import tensorflow as tf

from models.elmo import ELMO
from dataset import ElmoKoreanDataset


class ELMOTrainer:
    def __init__(self, config):
        self.config = config

        # define ELMO model and dataset
        self.train_dataset = ElmoKoreanDataset(config)
        self.steps_per_epoch = int(self.train_dataset.get_corpus_size() / config["batch_size"])
        self.elmo = ELMO(config)

        # init train operations
        self.optimizer = tf.train.AdamOptimizer()
        self.train_global_steps = 0

        self.train_iter, self.train_batch = self.input_data(self.train_dataset)
        self.train_loss, self.train_acc, self.train_ops = \
            self.elmo.train(self.train_batch, self.train_global_steps)

        # init ELMO variables & session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config["log_dir"] + '/train',
                                            self.sess.graph,
                                            filename_suffix=config["log_file_prefix"])

    def train(self, verbose=True):
        for epoch in range(self.config["epochs"]):
            self.sess.run(self.train_iter.initializer)

            for step in range(self.steps_per_epoch):
                try:
                    ops = [self.summary_merge, self.train_loss, self.train_acc, self.train_ops]
                    summary, loss, acc, _ = self.sess.run(ops)
                    self.writer.add_summary(summary, global_step=self.train_global_steps)

                    if verbose and step % self.config["verbose_freq"] == 0:
                        self.train_logging(epoch, step, loss, acc)

                    if self.train_global_steps % self.config["save_freq"] == 0:
                        self.saver.save(self.sess, save_path=self.config["model_save_path"],
                                        global_step=self.train_global_steps)

                    self.train_global_steps += 1

                except tf.errors.OutOfRangeError:
                    break

    def train_logging(self, epoch, step, loss, acc):
        print("Train EP:%d [%d/%d] loss: %f acc: %f" %
              (epoch, step, self.steps_per_epoch, loss.item(), acc.item()))

    def eval(self):
        pass

    def input_data(self, dataset):
        output_shapes = {"input": [self.config["word_seq_len"], self.config["char_seq_len"]],
                         "input_len": [], "target": [self.config["word_seq_len"]], "target_len": []}
        output_types = {"input": tf.int32, "input_len": tf.int32, "target": tf.int32, "target_len": tf.int32}

        tf_dataset = tf.data.Dataset.from_generator(dataset.data_generator,
                                                    output_shapes=output_shapes,
                                                    output_types=output_types)

        tf_dataset = tf_dataset.batch(self.config["batch_size"])
        tf_dataset = tf_dataset.prefetch(self.config["prefetch_size"])
        iterator = tf_dataset.make_initializable_iterator()
        batch_data_generator = iterator.get_next()
        return iterator, batch_data_generator
