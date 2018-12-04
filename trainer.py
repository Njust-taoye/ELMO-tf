import tensorflow as tf

from models.elmo import ELMO
from dataset import ElmoKoreanDataset


class ELMOTrainer:
    def __init__(self, config):
        self.config = config

        # define ELMO model and dataset
        self.train_dataset = ElmoKoreanDataset(config)
        self.elmo = ELMO(config)

        # init train operations
        self.optimizer = tf.train.AdamOptimizer()
        self.train_global_steps = tf.Variable(0)

        self.train_iter, self.train_batch = self.input_data(self.train_dataset)
        self.train_loss, self.train_ops = self.elmo.train(self.train_batch, self.train_global_steps)

        # init ELMO variables & session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, verbose=True):
        for epoch in range(self.config["epochs"]):
            self.sess.run(self.train_iter.initializer)

            for step in range(self.config["steps_per_epoch"]):
                try:
                    loss, _ = self.sess.run([self.train_loss, self.train_ops])

                    if verbose and step % self.config["verbose_freq"] == 0:
                        self.train_logging(epoch, step, loss)

                except tf.errors.OutOfRangeError:
                    break

    def train_logging(self, epoch, step, loss):
        print("Train EP:%d [%d/%d] loss: %f" % (epoch, step, self.config["steps_per_epoch"], loss.item()))

    def eval(self):
        pass

    def input_data(self, dataset):
        output_shapes = {"input": [self.config["word_seq_len"], self.config["char_seq_len"]],
                         "input_len": [], "target": [self.config["word_seq_len"]], "target_len": []}
        output_types = {"input": tf.int32, "input_len": tf.int32, "target": tf.int32, "target_len": tf.int32}

        tf_dataset = tf.data.Dataset.from_generator(dataset.data_generator,
                                                    output_shapes=output_shapes,
                                                    output_types=output_types)

        iterator = tf_dataset.batch(self.config["batch_size"]).make_initializable_iterator()
        batch_data_generator = iterator.get_next()
        return iterator, batch_data_generator
