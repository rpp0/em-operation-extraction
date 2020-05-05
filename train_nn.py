#!/usr/bin/env python
import sys
sys.path.append("./lib/fast-wavenet")
import matplotlib.pyplot as plt
import code
import argparse
import wavenet.utils

from wavenet.utils import get_wavenet_data, get_normalized_data
from wavenet.models import WavenetModel, ClassificationModel, BestCNN, BestCNNBB, get_cnnbb_loss
from emma.io.io import get_trace_set
from emma.processing.dsp import *
from common import *


filter_method = 'abs_nofilt'
datasets_root = './datasets/'
# input_size = 32768  # Test
input_size = 131072
#input_size = 4096
num_classes = len(op_to_int.keys())
noise_patch = None
use_augment_noise = False
use_newaugment = None  # Use argument


def label_to_title(label):
    label_class = np.argmax(label[0:num_classes])
    return int_to_op[label_class]


def augment_trace_older(trace, meta):
    if "left_bound" in meta:
        op_length = int(meta["right_bound"] - meta["left_bound"])
        new_position = np.random.randint(0, len(trace)-op_length)
        to_roll = int(meta["left_bound"] - new_position)
        new_trace = np.roll(trace, -to_roll)
        meta["left_bound"] = new_position
        meta["right_bound"] = new_position + op_length
        return new_trace
    else:
        print("No bounds; roll over entire length")
        return np.roll(trace, np.random.randint(0, len(trace)))


def augment_trace_old(trace, meta):
    if "left_bound" in meta:
        left = int(meta["left_bound"])
        right = int(meta["right_bound"])
        op_length = right - left

        # Shift to new position
        new_position = np.random.randint(0, len(trace)-op_length+1)
        if new_position > left:
            new_trace = np.concatenate((trace[0:left], trace[right:right+(new_position-left)], trace[left:right], trace[right+(new_position-left):]))
        else:
            new_trace = np.concatenate((trace[0:new_position], trace[left:right], trace[new_position:left], trace[right:]))
        assert(len(new_trace) == len(trace))

        # Add some noise
        if use_augment_noise:
            trace_mean = np.mean(new_trace)
            trace_std = np.std(new_trace)
            noise = np.random.normal(loc=trace_mean, scale=trace_std, size=len(new_trace))
            new_trace = new_trace + noise

        meta["left_bound"] = new_position
        meta["right_bound"] = new_position + op_length
        return new_trace
    else:
        print("No bounds; roll over entire length")
        new_trace = np.roll(trace, np.random.randint(0, len(trace)))

        if use_augment_noise:
            trace_mean = np.mean(new_trace)
            trace_std = np.std(new_trace)
            noise = np.random.normal(loc=trace_mean, scale=trace_std, size=len(new_trace))
            new_trace = new_trace + noise

        return new_trace


def augment_trace(trace, meta):
    if meta["op"] != "noise" and "left_bound" not in meta:
        print("No bounds; roll over entire length")
        new_trace = np.roll(trace, np.random.randint(0, len(trace)))

        if use_augment_noise:
            trace_mean = np.mean(new_trace)
            trace_std = np.std(new_trace)
            noise = np.random.normal(loc=trace_mean, scale=trace_std, size=len(new_trace))
            new_trace = new_trace + noise

        return new_trace
    else:
        print("Augmenting with noise snippet")
        if meta["op"] != "noise":
            left = int(meta["left_bound"])
            right = int(meta["right_bound"])
            op_length = right - left
        else:
            left = np.random.randint(0, len(trace))
            right = np.random.randint(left, len(trace))
            op_length = right - left

        # Shift to new position
        new_position = np.random.randint(0, len(trace)-op_length+1)

        num_pre_samples = new_position
        num_post_samples = len(trace)-(num_pre_samples + op_length)

        pre_samples_start = np.random.randint(0, len(noise_patch)-num_pre_samples)
        pre_samples = noise_patch[pre_samples_start:pre_samples_start+num_pre_samples]
        post_samples_start = np.random.randint(0, len(noise_patch)-num_post_samples)
        post_samples = noise_patch[post_samples_start:post_samples_start+num_post_samples]

        new_trace = np.concatenate((pre_samples, trace[left:right], post_samples))
        assert(len(new_trace) == len(trace))

        # Add some noise
        if use_augment_noise:
            trace_mean = np.mean(new_trace)
            trace_std = np.std(new_trace)
            noise = np.random.normal(loc=trace_mean, scale=trace_std, size=len(new_trace))
            new_trace = new_trace + noise

        if meta["op"] != "noise":
            meta["left_bound"] = new_position
            meta["right_bound"] = new_position + op_length
        return new_trace


def get_batch(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, add_bounds=False, augment=False, alwaysnew=False):
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
        filtered_trace = pad_to_length(trace.signal, input_size + 1)
        filtered_trace = filter_trace(filtered_trace, filter_method)
        #plt.plot(filtered_trace)
        #plot_meta(meta_trace_set[j])
        #plt.show()
        if augment:
            if np.random.randint(randint_min, randint_max) == 1:
                filtered_trace = augment_trace(filtered_trace, meta=meta_trace_set[j])
            else:
                filtered_trace = augment_trace_old(filtered_trace, meta=meta_trace_set[j])
        #plt.plot(filtered_trace)
        #plot_meta(meta_trace_set[j])
        #plt.show()

        if wavenet:
            wavenet_input, wavenet_target = get_wavenet_data(filtered_trace)
        else:
            wavenet_input, wavenet_target = get_normalized_data(filtered_trace)

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
            label.extend([mid / input_size, dist / input_size])
            label = np.array(label)
        batch_c.append((wavenet_input.flatten(), wavenet_target.flatten(), label))

        if len(batch_c) == batch_size:
            batch = np.array(batch_c)
            np.random.shuffle(batch)
            yield batch
            batch_c.clear()


def get_fullconnect_as_test_batch(dataset_path, dataset_file, batch_c, batch_size=1, wavenet=False, add_bounds=False, augment=False):
    # Get metadata and trace paths
    trace_path = os.path.join(dataset_path, dataset_file)

    # Get traces
    trace_set = get_trace_set(trace_path, 'cw', remote=False)

    batch_id = 0
    for j, trace in enumerate(trace_set.traces):
        for k in range(0, len(trace.signal), int(input_size/2)):
            if batch_id < 2311:  # TODO remove me! Debug
                batch_id += 1
                continue
            print("Batch %d window: [%d:%d]" % (batch_id, k, k+input_size+1))
            filtered_trace = pad_to_length(trace.signal[k:k+input_size+1], input_size+1)

            #if batch_id in [2311, 2331, 2335]:  # TODO remove me! debug
            debug_trace_specgram(filtered_trace, 56000000)

            filtered_trace = filter_trace(filtered_trace, filter_method)

            if wavenet:
                wavenet_input, wavenet_target = get_wavenet_data(filtered_trace)
            else:
                wavenet_input, wavenet_target = get_normalized_data(filtered_trace)

            batch_c.append((wavenet_input.flatten(), wavenet_target.flatten(), np.zeros(num_classes + 2)))
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
            for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=64, add_bounds=with_bounds, augment=False, wavenet=(model_to_use == "wavenet")):
                return batch


def train_datasets(dataset_names, load=False, with_bounds=False, epochs=1):
    try:
        validation_batch = get_validation_batch(dataset_names[0].replace("train", "val"), with_bounds=with_bounds)
    except Exception:
        print("Failed to get validation set")
        validation_batch = None

    if model_to_use == "wavenet":
        model = ClassificationModel(input_size, num_classes, num_layers=16, load=load)
    else:
        if with_bounds:
            model = BestCNNBB(input_size, num_classes, load=load, valbatch=validation_batch)
        else:
            model = BestCNN(input_size, num_classes, load=load, valbatch=validation_batch)
    print("Receptive field: %d" % model.calculate_receptive_field())
    # assert (input_size == model.calculate_receptive_field() + 1)  # If we want receptive size to match size of trace (not really required here)

    batch_c = []  # Batch container
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for epoch in range(epochs):
            for i, dataset_file in enumerate(dataset_files):
                if '_traces.npy' in dataset_file:
                    for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=batch_size, add_bounds=with_bounds, augment=with_bounds, wavenet=(model_to_use == "wavenet")):
                        model.train_batch(batch)


def test_datasets(dataset_names, with_bounds=False, augment=False):
    if model_to_use == "wavenet":
        model = ClassificationModel(input_size, num_classes, num_layers=16, load=True)
    else:
        if with_bounds:
            model = BestCNNBB(input_size, num_classes, load=True)
        else:
            model = BestCNN(input_size, num_classes, load=True)

    aug_name = "-aug" if augment else ""
    if with_bounds:
        confusion_matrix = ConfusionMatrix("seg-dl-%s-%s%s" % (filter_method, model_to_use, aug_name))
        distance_counter = DistanceCounter("seg-dl-%s-%s%s" % (filter_method, model_to_use, aug_name), num_classes, input_size)
    else:
        confusion_matrix = ConfusionMatrix("roi-dl-%s-%s%s" % (filter_method, model_to_use, aug_name))
        distance_counter = None

    batch_c = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_root, dataset_name)
        dataset_files = list(os.listdir(dataset_path))

        for i, dataset_file in enumerate(dataset_files):
            if '_traces.npy' in dataset_file:
                for batch in get_batch(dataset_path, dataset_file, batch_c, batch_size=1, add_bounds=with_bounds, augment=augment, wavenet=(model_to_use == "wavenet"), alwaysnew=augment):
                    predictions = model.test_batch(batch)
                    true_values = batch[:, 2]  # Get labels column of batch

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
                                mid = predicted_label[num_classes]*input_size
                                dist = predicted_label[num_classes+1]*input_size
                                fake_left = mid - (dist / 2.0)
                                fake_right = fake_left + dist
                                print((fake_left, fake_right))
                                fake_meta = {"left_bound": fake_left, "right_bound": fake_right}
                                #plt.title(label_to_title(predicted_label))
                                #if label_to_title(predicted_label) != "noise":
                                #    plot_meta(fake_meta)
                                #plt.plot(data)
                                #plt.show()
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
        model = BestCNNBB(input_size, num_classes, load=True)
    else:
        model = BestCNN(input_size, num_classes, load=True)
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
                        mid = predicted_label[num_classes]*input_size
                        dist = predicted_label[num_classes+1]*input_size
                        fake_left = mid - (dist / 2.0)
                        fake_right = fake_left + dist
                        print((fake_left, fake_right))
                        fake_meta = {"left_bound": fake_left, "right_bound": fake_right}
                        if label_to_title(predicted_label) != "noise":
                            plt.title(label_to_title(predicted_label))
                            plot_meta(fake_meta)
                            #plt.plot(butter_filter(data, 1, 0.01, 'low', None))
                            plt.plot(data)
                            plt.show()


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


parser = argparse.ArgumentParser(description="Wavenet.")
parser.add_argument("action", type=str, choices=["train", "test", "trainbb", "testbb", "testbbfullconnect", "testbbaugment", "qa"], help="Action to perform.")
parser.add_argument("model", type=str, choices=["wavenet", "best_cnn"], help="Model to use.")
parser.add_argument("filter_method", type=str, choices=["none", "ifreq", "abs_nofilt", "abs", "complex_hp"], help="Type of trace filtering to apply pre classification")
parser.add_argument("--datasets-root", type=str, default="./datasets/", help="Root of all datasets.")
parser.add_argument("--load", default=False, action="store_true", help="Reload.")
parser.add_argument("--use-newaugment", default=False, action="store_true", help="New augment.")
parser.add_argument("--epochs", type=int, default=1, help="Epochs.")
args = parser.parse_args()

# Args
filter_method = args.filter_method
datasets_root = args.datasets_root
batch_size = 20
model_to_use = args.model
if model_to_use == "wavenet":
    batch_size = 1

#noise_snippets = snippetize(np.load("./datasets/noise.npy"))
noise_snippets = snippetize(np.load("./datasets/nodemcu-fullconnect/2020-02-19_11-52-45_598201_traces.npy")[0], snippet_length=128)
noise_patch = filter_trace(noise_snippets, filter_method)
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
