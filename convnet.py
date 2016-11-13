import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import tensorflow as tf
import tensorflow.contrib.layers as layers


def main():
    data_helper = DataHelper()
    train_x, train_y = data_helper.load_data('dataset/train.csv', train=True)
    test_x = data_helper.load_data('dataset/test.csv')

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    print('Train size:', len(train_x))
    print('Validation size:', len(val_x))

    conv_net = TFConvNet(feature_num=784, class_num=10, is_training=False)
    conv_net.train(train_x, train_y, val_x, val_y, keep_prob=0.5)
    conv_net.generate_submission(test_x)


class DataHelper(object):

    @staticmethod
    def load_data(path, train=False):
        data = pd.read_csv(path)

        if train:
            images = data.iloc[:, 1:].values
        else:
            images = data.iloc[:, :].values

        images = images.astype(np.float)
        images = np.multiply(images, 1.0 / 255.0)
        image_size = images.shape[1]
        image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

        images = images.reshape([-1, 28, 28, 1])
        print(path, images[0].shape, len(images))

        if train:
            labels_flat = data[[0]].values.ravel()
            labels_count = np.unique(labels_flat).shape[0]
            labels = DataHelper.dense_to_one_hot(labels_flat, labels_count)
            labels = labels.astype(np.uint8)

            return images, labels

        return images

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


class TFConvNet(object):
    def __init__(self, feature_num, class_num, is_training, epochs=500000, batch_size=100, learning_rate=5e-4):
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = 1e-3

        self.bn_params = {
            # Decay for the moving averages.
            'decay': 0.999,
            'center': True,
            'scale': True,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # None to force the updates during train_op
            'updates_collections': None,
            'is_training': is_training
        }

        self.feature_num = feature_num
        self.class_num = class_num

        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, feature_num])
        self.y_ = tf.placeholder(tf.float32, [None, class_num])

        with tf.contrib.framework.arg_scope(
                [layers.convolution2d],
                kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(self.weight_decay)
        ):
            self.X = tf.reshape(self.X, [-1, 28, 28, 1])

            net = layers.convolution2d(self.X, num_outputs=16)
            net = layers.convolution2d(net, num_outputs=16)
            net = layers.max_pool2d(net, kernel_size=2)
            net = layers.relu(net, num_outputs=16)

            net = layers.convolution2d(net, num_outputs=32)
            net = layers.convolution2d(net, num_outputs=32)
            net = layers.max_pool2d(net, kernel_size=2)
            net = layers.dropout(net, keep_prob=self.keep_prob)
            net = layers.relu(net, num_outputs=32)

            net = layers.flatten(net, [-1, 7 * 7 * 32])
            net = layers.fully_connected(net, num_outputs=64, activation_fn=tf.nn.tanh)
            net = layers.dropout(net, keep_prob=self.keep_prob)

            net = layers.fully_connected(net, num_outputs=self.class_num, activation_fn=tf.nn.tanh)
            self.y = layers.softmax(net)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(pred, tf.float32))

        self.sess = tf.Session()

    def train(self, X_train, y_train, X_val, y_val, keep_prob=0.5):
        print("\nStarting to train\n")
        self.sess.run(tf.initialize_all_variables())

        batch_start = 0
        batch_end = batch_start + self.batch_size
        for iteration in range(self.epochs):
            _, loss, probs = self.sess.run(
                [self.optimizer, self.loss, self.y],
                feed_dict={
                    self.X: X_train[batch_start:batch_end],
                    self.y_: y_train[batch_start:batch_end],
                    self.keep_prob: keep_prob
                }
            )

            if iteration % 50 == 0:
                train_acc = self.sess.run(self.acc, feed_dict={
                    self.X: X_train[batch_start:batch_end],
                    self.y_: y_train[batch_start:batch_end],
                    self.keep_prob: 1.0
                })

                val_acc = self.sess.run(
                    self.acc,
                    feed_dict={self.X: X_val, self.y_: y_val, self.keep_prob: 1.0}
                )

                print('Iteration: {}, loss: {:2.4}, train acc: {:.3%}, validation acc: {:.3%}'.format(
                    iteration, loss, train_acc, val_acc))

                if val_acc >= 0.991:
                    print('Validation acc good! Breaking!')
                    break

            batch_start = batch_end
            batch_end += self.batch_size

            if batch_end > len(X_train):
                batch_start = 0
                batch_end = batch_start + self.batch_size
                X_train, y_train = self.__shuffle(X_train, y_train)

        print("\nTraining ended")

    def __shuffle(self, a, b):
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def generate_submission(self, test_x, file_name='submission.csv'):
        print('Preparing to generate submission.csv')
        predict = tf.argmax(self.y, 1)

        predictions = []
        for i in range(0, test_x.shape[0] // self.batch_size):
            predict_batch = self.sess.run(
                [predict], feed_dict={
                    self.X: test_x[i * self.batch_size: (i + 1) * self.batch_size],
                    self.keep_prob: 1.0
                }
            )

            predictions.extend(list(predict_batch[0]))

        np.savetxt(
            file_name, np.c_[range(1, len(test_x) + 1), predictions],
            delimiter=',', header='ImageId,Label', comments='', fmt='%d'
        )

        print('saved: %s' % file_name)


if __name__ == '__main__':
    main()
