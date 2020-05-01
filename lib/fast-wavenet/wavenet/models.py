import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.models
import keras.layers
import keras.activations
import os
import pickle
import keras.backend as K
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, AveragePooling1D, LeakyReLU, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Model, Sequential
from .layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d, dense_relu)


class WavenetModel(object):
    def __init__(self,
                 num_time_samples,
                 num_channels=1,
                 num_bins=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 gpu_fraction=1.0):

        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction

        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        outputs = conv1d(h,
                         num_bins,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            outputs, targets)
        cost = tf.reduce_mean(costs)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = hs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost = self._train(inputs, targets)
            if cost < 1e-1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()


class ClassificationModel(object):
    def __init__(self,
                 num_time_samples,
                 num_classes,
                 num_channels=1,
                 num_bins=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 gpu_fraction=1.0,
                 filter_width=2,
                 load=False):

        # Disable GPU
        if True:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            data_format = 'NWC'
        else:
            data_format = 'NHWC'
        # --------------------
        self.num_time_samples = num_time_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction
        self.filter_width = filter_width
        self.dilations = []

        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))
        labels = tf.placeholder(tf.float32, shape=(None, num_classes))

        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                self.dilations.append(rate)
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name, filter_width=filter_width)
                hs.append(h)

        outputs = conv1d(h,
                         num_bins,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        # Train targets
        wavenet_costs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
        wavenet_cost = tf.reduce_mean(wavenet_costs)

        # Train classes
        print(outputs.shape)
        outputs = tf.nn.avg_pool1d(outputs, ksize=[1, 128, 1], strides=[1, 128, 1], padding='VALID', data_format=data_format)
        print(outputs.shape)
        outputs = conv1d(outputs, num_bins, filter_width=7, name="post1")
        print(outputs.shape)
        outputs = tf.nn.max_pool1d(outputs, ksize=[1, 4, 1], strides=[1, 4, 1], padding='VALID', data_format=data_format)
        print(outputs.shape)
        outputs = conv1d(outputs, num_bins, filter_width=5, name="post2")
        print(outputs.shape)
        outputs = tf.nn.max_pool1d(outputs, ksize=[1, 2, 1], strides=[1, 2, 1], padding='VALID', data_format=data_format)
        print(outputs.shape)
        outputs = conv1d(outputs, num_bins, filter_width=3, name="post3")
        print(outputs.shape)
        outputs = tf.nn.max_pool1d(outputs, ksize=[1, 2, 1], strides=[1, 2, 1], padding='VALID', data_format=data_format)
        print(outputs.shape)
        outputs = conv1d(outputs, num_bins, filter_width=3, name="post4")
        print(outputs.shape)
        outputs = tf.nn.max_pool1d(outputs, ksize=[1, 2, 1], strides=[1, 2, 1], padding='VALID', data_format=data_format)
        print(outputs.shape)
        outputs = tf.reshape(outputs, [-1, outputs.shape[-2] * outputs.shape[-1]])
        print(outputs.shape)
        outputs = dense_relu(outputs, int(outputs.shape[-1]), num_classes)
        print(outputs.shape)
        classification_costs = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
        classification_cost = tf.reduce_mean(classification_costs)

        alpha = 0.7
        cost = (alpha * classification_cost) + ((1.0-alpha) * wavenet_cost)

        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        # Add ops to save and restore all the variables.
        self.save_path = "/tmp/wavenet.ckpt"
        self.saver = tf.train.Saver()
        if load:
            self.saver.restore(sess, self.save_path)

        self.inputs = inputs
        self.targets = targets
        self.labels = labels
        self.outputs = outputs
        self.hs = hs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess
        self.loss_history = []

    def calculate_receptive_field(self):
        receptive_field = (self.filter_width - 1) * sum(self.dilations) + 1
        return receptive_field

    def _train(self, inputs, targets, labels):
        feed_dict = {self.inputs: inputs, self.targets: targets, self.labels: labels}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets, labels):
        cost = self._train(inputs, targets, labels)
        print(cost)
        self.loss_history.append(cost)

        # Save
        self.saver.save(self.sess, self.save_path)

    def test(self, inputs):
        feed_dict = {self.inputs: inputs}
        outputs = self.sess.run(
            [tf.nn.softmax(self.outputs)],
            feed_dict=feed_dict)
        return outputs

    def _split_batch(self, batch):
        inputs = np.stack(batch[:,0], axis=0)[:, :, None]
        targets = np.stack(batch[:,1], axis=0)
        labels = np.stack(batch[:,2], axis=0)
        assert(inputs.shape[1] == self.num_time_samples)  # Samples
        assert(inputs.shape[2] == 1)  # Channels
        assert(targets.shape[1] == self.num_time_samples)
        assert(labels.shape[1] == self.num_classes)
        return inputs, targets, labels

    def train_batch(self, batch):
        print("Training on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)
        self.train(inputs, targets, labels)

    def test_batch(self, batch):
        print("Testing on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)
        return self.test(inputs)


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, self.model.num_bins)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden

                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops

        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_


class BestCNN:
    def __init__(self, num_inputs, num_classes, load=False, valbatch=None):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.input_shape = (num_inputs, 1)
        self.valbatch = valbatch
        self.lowest_loss = None
        self.loss_history = []
        if load:
            self.model = keras.models.load_model('/tmp/best_cnn.h5')
        else:
            self.model = self._get_model()

    def _get_model(self):
        # From VGG16 design
        img_input = Input(shape=self.input_shape)
        # Block 1
        x = Conv1D(16, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        # Block 2
        x = Conv1D(32, 11, activation='relu', padding='same', name='block2_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        # Block 3
        x = Conv1D(64, 11, activation='relu', padding='same', name='block3_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
        # Block 4
        x = Conv1D(128, 11, activation='relu', padding='same', name='block4_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
        # Block 5
        x = Conv1D(128, 11, activation='relu', padding='same', name='block5_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
        # Block 6
        x = Conv1D(128, 11, activation='relu', padding='same', name='block6_conv1')(x)
        x = AveragePooling1D(4, strides=4, name='block6_pool')(x)
        # Block 7
        x = Conv1D(128, 11, activation='relu', padding='same', name='block7_conv1')(x)
        x = AveragePooling1D(4, strides=4, name='block7_pool')(x)
        # Block 8
        x = Conv1D(128, 11, activation='relu', padding='same', name='block8_conv1')(x)
        x = AveragePooling1D(4, strides=4, name='block8_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', name='fc1')(x)
        x = Dense(2048, activation='relu', name='fc2')(x)
        x = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        inputs = img_input
        # Create model.
        model = Model(inputs, x, name='cnn_best')
        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def _split_batch(self, batch):
        inputs = np.stack(batch[:,0], axis=0)[:, :, None]
        targets = np.stack(batch[:,1], axis=0)
        labels = np.stack(batch[:,2], axis=0)
        assert(inputs.shape[1] == self.num_inputs)  # Samples
        assert(inputs.shape[2] == 1)  # Channels
        assert(targets.shape[1] == self.num_inputs)
        assert(labels.shape[1] == self.num_classes)
        return inputs, targets, labels

    def calculate_receptive_field(self):
        return 0

    def save_if_lower_loss(self, val_result):
        current_loss = val_result[0]
        self.loss_history.append(current_loss)
        with open("/tmp/best_cnn-low.p", "wb") as f:
            pickle.dump(self.loss_history, f)
        if self.lowest_loss is None:
            self.lowest_loss = current_loss

        print("Current val loss is %f (lowest %f)" % (current_loss, self.lowest_loss))

        if current_loss <= self.lowest_loss:
            self.lowest_loss = current_loss
            self.model.save('/tmp/best_cnn-low.h5')

    def train_batch(self, batch):
        print("Training on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        loss = self.model.train_on_batch(inputs, labels)
        print(loss)
        self.model.save('/tmp/best_cnn.h5')

        if self.valbatch is not None:
            val_result = self.evaluate_batch(self.valbatch)
            self.save_if_lower_loss(val_result)

    def test_batch(self, batch):
        print("Testing on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        return self.model.predict(inputs)

    def evaluate_batch(self, batch):
        print("Evaluating on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        return self.model.evaluate(inputs, labels)

def get_cnnbb_loss(num_classes):
    def cnnbb_loss(y_true, y_predicted):
        is_noise = y_true[:,num_classes-1] # Assume last class is noise
        noise_weight = 1 - is_noise
        loss = 0
        loss += K.categorical_crossentropy(y_true[:,0:num_classes], y_predicted[:,0:num_classes], from_logits=False)

        true_mid = y_true[:,num_classes]
        pred_mid = y_predicted[:,num_classes]

        true_dist = y_true[:,num_classes+1]
        pred_dist = y_predicted[:,num_classes+1]

        loss += 10 * noise_weight * K.square((true_mid - pred_mid))
        loss += 10 * noise_weight * K.square((true_dist - pred_dist))

        return K.mean(loss)
    return cnnbb_loss

def get_cnnbb_acc(num_classes):
    def cnnbb_acc(y_true, y_pred):
        return K.cast(K.equal(
                    K.argmax(y_true[:, 0:num_classes], axis=-1),
                    K.argmax(y_pred[:, 0:num_classes], axis=-1)),
                K.floatx())
    return cnnbb_acc

class BestCNNBB:
    def __init__(self, num_inputs, num_classes, load=False, valbatch=None):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.input_shape = (num_inputs, 1)
        self.valbatch = valbatch
        self.lowest_loss = None
        self.loss_history = []
        self.custom_objects = {
            'cnnbb_acc': get_cnnbb_acc(num_classes),
            'cnnbb_loss': get_cnnbb_loss(num_classes),
        }
        if load:
            self.model = keras.models.load_model('/tmp/best_cnn_bb.h5', custom_objects=self.custom_objects)
        else:
            self.model = keras.models.load_model('/tmp/best_cnn.h5')
            self.model = self._update_model()

    def _update_model(self):
        # Cut off last layer and connect it to new dense layer
        x = Dense(self.num_classes + 2, activation=None, name='predictions')(self.model.layers[-2].output)

        #
        lambda_self_bug = self.num_classes
        x_classes = keras.layers.Lambda(lambda l: l[:,0:lambda_self_bug])(x)
        x_bounds = keras.layers.Lambda(lambda l: l[:,lambda_self_bug:lambda_self_bug+2])(x)
        classes_out = keras.layers.Activation('softmax')(x_classes)
        bounds_out = keras.layers.Activation('relu')(x_bounds)
        x = keras.layers.Concatenate(axis=-1)([classes_out, bounds_out])

        # Create model.
        inputs = self.model.inputs
        model = Model(inputs, x, name='cnn_best_bb')
        print(model.summary())
        optimizer = Adam(lr=0.0001)
        model.compile(loss=get_cnnbb_loss(self.num_classes), optimizer=optimizer, metrics=[get_cnnbb_acc(self.num_classes)])
        return model

    def _split_batch(self, batch):
        inputs = np.stack(batch[:,0], axis=0)[:, :, None]
        targets = np.stack(batch[:,1], axis=0)
        labels = np.stack(batch[:,2], axis=0)
        assert(inputs.shape[1] == self.num_inputs)  # Samples
        assert(inputs.shape[2] == 1)  # Channels
        assert(targets.shape[1] == self.num_inputs)
        assert(labels.shape[1] == self.num_classes+2)
        return inputs, targets, labels

    def calculate_receptive_field(self):
        return 0

    def save_if_lower_loss(self, val_result):
        current_loss = val_result[0]
        self.loss_history.append(current_loss)
        with open("/tmp/best_cnn_bb-low.p", "wb") as f:
            pickle.dump(self.loss_history, f)
        if self.lowest_loss is None:
            self.lowest_loss = current_loss

        print("Current val loss is %f (lowest %f)" % (current_loss, self.lowest_loss))

        if current_loss <= self.lowest_loss:
            self.lowest_loss = current_loss
            self.model.save('/tmp/best_cnn_bb-low.h5')

    def train_batch(self, batch):
        print("Training on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        loss = self.model.train_on_batch(inputs, labels)
        print(loss)
        self.model.save('/tmp/best_cnn_bb.h5')

        if self.valbatch is not None:
            val_result = self.evaluate_batch(self.valbatch)
            self.save_if_lower_loss(val_result)


    def test_batch(self, batch):
        print("Testing on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        return self.model.predict(inputs)

    def evaluate_batch(self, batch):
        print("Evaluating on batch of %d traces" % batch.shape[0])
        inputs, targets, labels = self._split_batch(batch)

        return self.model.evaluate(inputs, labels)


class BestCNNBB2D:
    def __init__(self, num_inputs, num_classes, load=False, valbatch=None):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.input_shape = (*num_inputs, 1)
        self.valbatch = valbatch
        self.lowest_loss = None
        self.loss_history = []
        self.custom_objects = {
            'cnnbb_acc': get_cnnbb_acc(num_classes),
            'cnnbb_loss': get_cnnbb_loss(num_classes),
        }
        if load:
            self.model = keras.models.load_model('/tmp/best_cnn_bb_2d.h5', custom_objects=self.custom_objects)
        else:
            self.model = self._get_model()

    def _get_model(self):
        # From VGG16 design
        inputs = Input(shape=self.input_shape)
        # Block 1
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        # Block 2
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        # Block 3
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        # Block 4
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # Block 5
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', name='fc1')(x)
        x = Dense(2048, activation='relu', name='fc2')(x)
        x = Dense(self.num_classes+2, activation=None, name='predictions')(x)

        lambda_self_bug = self.num_classes
        x_classes = keras.layers.Lambda(lambda l: l[:,0:lambda_self_bug])(x)
        x_bounds = keras.layers.Lambda(lambda l: l[:,lambda_self_bug:lambda_self_bug+2])(x)
        classes_out = keras.layers.Activation('softmax')(x_classes)
        bounds_out = keras.layers.Activation('relu')(x_bounds)
        x = keras.layers.Concatenate(axis=-1)([classes_out, bounds_out])

        # Create model.
        model = Model(inputs, x, name='cnn2d_best')
        print(model.summary())
        optimizer = Adam(lr=0.0001)
        model.compile(loss=get_cnnbb_loss(self.num_classes), optimizer=optimizer, metrics=[get_cnnbb_acc(self.num_classes)])
        return model

    def _split_batch(self, batch):
        inputs = np.stack(batch[:,0], axis=0)[:, :, :, None]
        labels = np.stack(batch[:,1], axis=0)
        assert(inputs.shape[1:3] == self.num_inputs)  # Samples
        assert(inputs.shape[3] == 1)  # Channels
        assert(labels.shape[1] == self.num_classes+2)
        return inputs, labels

    def calculate_receptive_field(self):
        return 0

    def save_if_lower_loss(self, val_result):
        current_loss = val_result[0]
        self.loss_history.append(current_loss)
        with open("/tmp/best_cnn_bb_2d-low.p", "wb") as f:
            pickle.dump(self.loss_history, f)
        if self.lowest_loss is None:
            self.lowest_loss = current_loss

        print("Current val loss is %f (lowest %f)" % (current_loss, self.lowest_loss))

        if current_loss <= self.lowest_loss:
            self.lowest_loss = current_loss
            self.model.save('/tmp/best_cnn_bb_2d-low.h5')

    def train_batch(self, batch):
        print("Training on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        loss = self.model.train_on_batch(inputs, labels)
        print(loss)
        self.model.save('/tmp/best_cnn_bb_2d.h5')

        if self.valbatch is not None:
            val_result = self.evaluate_batch(self.valbatch)
            self.save_if_lower_loss(val_result)

    def test_batch(self, batch):
        print("Testing on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        return self.model.predict(inputs)

    def evaluate_batch(self, batch):
        print("Evaluating on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        return self.model.evaluate(inputs, labels)


class BestCNN2D:
    def __init__(self, num_inputs, num_classes, load=False, valbatch=None):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.input_shape = (*num_inputs, 1)
        self.valbatch = valbatch
        self.lowest_loss = None
        self.loss_history = []
        if load:
            self.model = keras.models.load_model('/tmp/best_cnn_2d.h5')
        else:
            self.model = self._get_model()

    def _get_model(self):
        inputs = Input(shape=self.input_shape)
        # Block 1
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        # Block 2
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        # Block 3
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        # Block 4
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # Block 5
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', name='fc1')(x)
        x = Dense(2048, activation='relu', name='fc2')(x)
        x = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(inputs, x, name='cnn_best_2d')
        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def _split_batch(self, batch):
        inputs = np.stack(batch[:,0], axis=0)[:, :, :, None]
        labels = np.stack(batch[:,1], axis=0)
        assert(inputs.shape[1:3] == self.num_inputs)  # Samples
        assert(inputs.shape[3] == 1)  # Channels
        assert(labels.shape[1] == self.num_classes)
        return inputs, labels

    def calculate_receptive_field(self):
        return 0

    def save_if_lower_loss(self, val_result):
        current_loss = val_result[0]
        self.loss_history.append(current_loss)
        with open("/tmp/best_cnn_2d-low.p", "wb") as f:
            pickle.dump(self.loss_history, f)
        if self.lowest_loss is None:
            self.lowest_loss = current_loss

        print("Current val loss is %f (lowest %f)" % (current_loss, self.lowest_loss))

        if current_loss <= self.lowest_loss:
            self.lowest_loss = current_loss
            self.model.save('/tmp/best_cnn_2d-low.h5')

    def train_batch(self, batch):
        print("Training on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        loss = self.model.train_on_batch(inputs, labels)
        print(loss)
        self.model.save('/tmp/best_cnn_2d.h5')

        if self.valbatch is not None:
            val_result = self.evaluate_batch(self.valbatch)
            self.save_if_lower_loss(val_result)

    def test_batch(self, batch):
        print("Testing on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        return self.model.predict(inputs)

    def evaluate_batch(self, batch):
        print("Evaluating on batch of %d traces" % batch.shape[0])
        inputs, labels = self._split_batch(batch)

        return self.model.evaluate(inputs, labels)
