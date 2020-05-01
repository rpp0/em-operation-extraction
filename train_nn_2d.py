#!/usr/bin/env python
import sys
sys.path.append("./lib/fast-wavenet")
import matplotlib.pyplot as plt
import code
import argparse
import wavenet.utils
import sklearn.preprocessing

from wavenet.utils import get_wavenet_data
from wavenet.models import BestCNNBB2D, BestCNN2D, get_cnnbb_loss
from emma.io.io import get_trace_set
from emma.processing.dsp import *
from common import *


datasets_root = './datasets/'
window_size = 131072+256
num_stft_segments = 512
input_size = (num_stft_segments, num_stft_segments)
num_classes = len(op_to_int.keys())
noise_patch = None
use_newaugment = None  # Use argument


def get_normalized_data(trace):
    return sklearn.preprocessing.normalize(trace)


def label_to_title(label):
    label_class = np.argmax(label[0:num_classes])
    return int_to_op[label_class]


def transform_meta_1d_to_2d(meta, overlap=0.5):
    if "left_bound" in meta:
        meta["left_bound"] = (meta["left_bound"] / (512 * (1.0 - overlap)))
        meta["right_bound"] = (meta["right_bound"] / (512 * (1.0 - overlap)))
    return meta


def augment_trace_old(trace, meta):
    if "left_bound" in meta:
        left = int(meta["left_bound"])
        right = int(meta["right_bound"])
        op_length = right - left

        # Shift to new position
        new_position = np.random.randint(0, trace.shape[1]-op_length+1)
        if new_position > left:
            new_trace = np.concatenate((trace[:,0:left], trace[:,right:right+(new_position-left)], trace[:,left:right], trace[:,right+(new_position-left):]), axis=1)
        else:
            new_trace = np.concatenate((trace[:,0:new_position], trace[:,left:right], trace[:,new_position:left], trace[:,right:]), axis=1)
        assert(new_trace.shape == trace.shape)

        meta["left_bound"] = new_position
        meta["right_bound"] = new_position + op_length
        return new_trace
    else:
        print("No bounds; roll over entire length")
        new_trace = np.roll(trace, np.random.randint(0, trace.shape[1]), axis=1)

        return new_trace


def augment_trace(trace, meta):
    if meta["op"] != "noise" and "left_bound" not in meta:
        print("No bounds; roll over entire length")
        new_trace = np.roll(trace, np.random.randint(0, trace.shape[1]), axis=1)

        return new_trace
    else:
        print("Augmenting with noise snippet")
        if meta["op"] != "noise":
            left = int(meta["left_bound"])
            right = int(meta["right_bound"])
            op_length = right - left
        else:
            left = np.random.randint(0, trace.shape[1])
            right = np.random.randint(left, trace.shape[1])
            op_length = right - left

        # Shift to new position
        new_position = np.random.randint(0, trace.shape[1]-op_length+1)

        num_pre_samples = new_position
        num_post_samples = trace.shape[1]-(num_pre_samples + op_length)

        pre_samples_start = np.random.randint(0, noise_patch.shape[1]-num_pre_samples)
        pre_samples = noise_patch[:, pre_samples_start:pre_samples_start+num_pre_samples]
        post_samples_start = np.random.randint(0, noise_patch.shape[1]-num_post_samples)
        post_samples = noise_patch[:, post_samples_start:post_samples_start+num_post_samples]

        new_trace = np.concatenate((pre_samples, trace[:, left:right], post_samples), axis=1)
        assert(new_trace.shape == trace.shape)

        if meta["op"] != "noise":
            meta["left_bound"] = new_position
            meta["right_bound"] = new_position + op_length
        return new_trace


def get_batch(dataset_path, dataset_file, batch_c, batch_size=1, add_bounds=False, augment=False, alwaysnew=False):
    # Get metadata and trace paths
    trace_name = dataset_file.rpartition('_traces.npy')[0]
    meta_name = trace_name + "_meta.p"
    trace_path = os.path.join(dataset_path, dataset_file)
    meta_path = os.path.join(dataset_path, meta_name)

    # Get traces
    trace_set = get_trace_set(trace_path, 'cw', remote=False)

    # Get trace metadata
    meta_trace_set = load_meta(meta_path)

    randint_max = 2 if use_newaugment else 1  # 50% probability to use new augment if enabled
    randint_min = 1 if alwaysnew else 0

    for j, trace in enumerate(trace_set.traces):
        if "bad" in meta_trace_set[j]:  # Skip traces explicitly labeled as "bad"
            continue
        filtered_trace = pad_to_length(trace.signal, window_size)
        filtered_trace = get_stft(filtered_trace, overlap_rate=0.5, show_plot=False)
        meta_trace_set[j] = transform_meta_1d_to_2d(meta_trace_set[j], overlap=0.5)
        #plot_stft(filtered_trace, meta=meta_trace_set[j])
        if augment:
            if np.random.randint(randint_min, randint_max) == 1:
                filtered_trace = augment_trace(filtered_trace, meta=meta_trace_set[j])
            else:
                filtered_trace = augment_trace_old(filtered_trace, meta=meta_trace_set[j])
        #plot_stft(filtered_trace, meta=meta_trace_set[j])

        #plot_stft(filtered_trace, meta=meta_trace_set[j])
        nn_input = get_normalized_data(filtered_trace)
        #plot_stft(nn_input, meta=meta_trace_set[j])

        label = get_onehot(meta_trace_set[j]["op"])
        if add_bounds:
            try:
                left_bound = meta_trace_set[j]["left_bound"]
                right_bound = meta_trace_set[j]["right_bound"]
            except KeyError:
                left_bound = 0
                right_bound = 0
                print("Warning: no bounds for label %s" % meta_trace_set[j]["op"])
            label = list(label)
            dist = right_bound - left_bound
            mid = left_bound + (dist / 2.0)
            label.extend([mid / num_stft_segments, dist / num_stft_segments])
            label = np.array(label)
        batch_c.append((nn_input, label))

        if len(batch_c) == batch_size:
            batch = np.array(batch_c)
            np.random.shuffle(batch)
            yield batch
            batch_c.clear()


def get_fullconnect_as_test_batch(dataset_path, dataset_file, batch_c, batch_size=1, add_bounds=False, augment=False):
    # Get metadata and trace paths
    trace_path = os.path.join(dataset_path, dataset_file)

    # Get traces
    trace_set = get_trace_set(trace_path, 'cw', remote=False)

    batch_id = 0
    for j, trace in enumerate(trace_set.traces):
        for k in range(0, len(trace.signal), int(window_size/2)):
            if batch_id < 2311:  # TODO remove me! Debug
                batch_id += 1
                continue
            print("Batch %d window: [%d:%d]" % (batch_id, k, k+window_size))
            filtered_trace = pad_to_length(trace.signal[k:k+window_size], window_size)

            filtered_trace = get_stft(filtered_trace, overlap_rate=0.5, show_plot=False)
            #plot_stft(filtered_trace)

            nn_input = get_normalized_data(filtered_trace)

            batch_c.append((nn_input, np.zeros(num_classes + 2)))
            batch_id += 1

            if len(batch_c) == batch_size:
                batch = np.array(batch_c)
                yield batch
                batch_c.clear()


def get_validation_batch(dataset_name, with_bounds=False):
    dataset_path = os.path.join(datasets_root, dataset_name)
    dataset_files = list(os.listdir(dataset_path))

    batch_c = []
    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=64, add_bounds=with_bounds, augment=False):
                return batch


def train_datasets(dataset_names, load=False, with_bounds=False, epochs=1):
    try:
        validation_batch = get_validation_batch(dataset_names[0].replace("train", "val"), with_bounds=with_bounds)
    except Exception:
        print("Failed to get validation set")
        validation_batch = None

    if with_bounds:
        model = BestCNNBB2D(input_size, num_classes, load=load, valbatch=validation_batch)
    else:
        model = BestCNN2D(input_size, num_classes, load=load, valbatch=validation_batch)
    print("Receptive field: %d" % model.calculate_receptive_field())

    batch_c = []  # Batch container
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for epoch in range(epochs):
            for i, dataset_file in enumerate(dataset_files):
                if '_traces.npy' in dataset_file:
                    for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=batch_size, add_bounds=with_bounds, augment=with_bounds):
                        model.train_batch(batch)


def test_datasets(dataset_names, with_bounds=False, augment=False):
    if with_bounds:
        model = BestCNNBB2D(input_size, num_classes, load=True)
    else:
        model = BestCNN2D(input_size, num_classes, load=True)

    aug_name = "-aug" if augment else ""
    if with_bounds:
        confusion_matrix = ConfusionMatrix("seg-dl2d%s" % aug_name)
        distance_counter = DistanceCounter("seg-dl2d%s" % aug_name, num_classes, 131072)
    else:
        confusion_matrix = ConfusionMatrix("roi-dl2d%s" % aug_name)
        distance_counter = None

    batch_c = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataset_files):
            if '_traces.npy' in dataset_file:
                for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=1, add_bounds=with_bounds, augment=augment, alwaysnew=augment):
                    predictions = model.test_batch(batch)
                    true_values = batch[:, 1]  # Get labels column of batch

                    assert (true_values.shape[0] == predictions.shape[0])
                    for j in range(predictions.shape[0]):
                        true_label = true_values[j]
                        predicted_label = predictions[j]
                        assert (len(true_label) == len(predicted_label))

                        if with_bounds:
                            if true_label[num_classes-1] != 1:
                                # Plot test
                                data = batch[j, 0]
                                print(predicted_label)
                                mid = predicted_label[num_classes]*num_stft_segments
                                dist = predicted_label[num_classes+1]*num_stft_segments
                                fake_left = mid - (dist / 2.0)
                                fake_right = fake_left + dist
                                print((fake_left, fake_right))
                                fake_meta = {"left_bound": fake_left, "right_bound": fake_right}
                                #plt.title(label_to_title(predicted_label))
                                #plot_stft(data, meta=fake_meta)
                            else:
                                print("Skipping noise trace visualization")
                            confusion_matrix.add_onehot(true_label[0:num_classes], predicted_label[0:num_classes])
                            distance_counter.add_value(true_label, predicted_label)
                        else:
                            confusion_matrix.add_onehot(true_label, predicted_label)

    if with_bounds:
        distance_counter.save()
        distance_counter.print()
    confusion_matrix.save()
    confusion_matrix.print()


def test_fullconnect(dataset_names, with_bounds=False):
    if with_bounds:
        model = BestCNNBB2D(input_size, num_classes, load=True)
    else:
        model = BestCNN2D(input_size, num_classes, load=True)
    print(model.model.summary())

    batch_c = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataset_files):
            if '_traces.npy' in dataset_file:
                for batch in get_fullconnect_as_test_batch(dataset_path, dataset_file, batch_c, batch_size=1, add_bounds=with_bounds, augment=False):
                    predictions = model.test_batch(batch)

                    for j in range(predictions.shape[0]):
                        predicted_label = predictions[j]

                        # Plot test
                        data = batch[j, 0]
                        print(predicted_label)
                        mid = predicted_label[num_classes]*num_stft_segments
                        dist = predicted_label[num_classes+1]*num_stft_segments
                        fake_left = mid - (dist / 2.0)
                        fake_right = fake_left + dist
                        print((fake_left, fake_right))
                        fake_meta = {"left_bound": fake_left, "right_bound": fake_right}
                        if label_to_title(predicted_label) != "noise":
                            plt.title(label_to_title(predicted_label))
                            plot_stft(data, meta=fake_meta)


def qa_experiment():
    import keras
    import keras.backend as K

    # Augment QA
    trace = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    meta = {"left_bound": 4, "right_bound": 8}
    plot_meta(meta)
    plt.plot(trace)
    plt.show()
    trace = augment_trace(trace, meta)
    plot_meta(meta)
    plt.plot(trace)
    plt.show()

    # Keras loss qa
    true_in = keras.Input(shape=(4,))
    pred_in = keras.Input(shape=(4,))
    loss_func = get_cnnbb_loss(2)
    print(loss_func)

    eval_loss = K.function([true_in, pred_in], [loss_func(true_in, pred_in)])
    print(eval_loss([
        [[1, 0, 0.5, 1.0], [0, 1, 0.5, 1.0], [1, 0, 0.5, 1.0]],
        [[1, 0, 0.5, 1.0], [0, 1, 0.0, 0.0], [1, 0, 0.4, 1.0]]
    ]))


parser = argparse.ArgumentParser(description="")
parser.add_argument("action", type=str, choices=["train", "test", "trainbb", "testbb", "testbbfullconnect", "testbbaugment", "qa"], help="Action to perform.")
parser.add_argument("--datasets-root", type=str, default="./datasets/", help="Root of all datasets.")
parser.add_argument("--load", default=False, action="store_true", help="Reload.")
parser.add_argument("--use-newaugment", default=False, action="store_true", help="New augment.")
parser.add_argument("--epochs", type=int, default=1, help="Epochs.")
args = parser.parse_args()

# Args
datasets_root = args.datasets_root
batch_size = 20
#noise_snippets = snippetize(np.load("./datasets/noise.npy"))
noise_snippets = snippetize(np.load("./datasets/nodemcu-fullconnect/2020-02-19_11-52-45_598201_traces.npy")[0], snippet_length=512)
noise_patch = get_stft(noise_snippets, overlap_rate=0.5, show_plot=False)
use_newaugment = args.use_newaugment

# Actions
if args.action == "train":
    train_datasets(["nodemcu-random-train2"], load=args.load)
elif args.action == "test":
    test_datasets(["nodemcu-random-test2"])
elif args.action == "trainbb":
    train_datasets(["nodemcu-random-label-train"], load=args.load, with_bounds=True, epochs=args.epochs)
elif args.action == "testbb":
    test_datasets(["nodemcu-random-label-test"], with_bounds=True)
elif args.action == "testbbfullconnect":
    test_fullconnect(["nodemcu-fullconnect"], with_bounds=True)
elif args.action == "testbbaugment":
    test_datasets(["nodemcu-random-label-test"], with_bounds=True, augment=True)
elif args.action == "qa":
    qa_experiment()
