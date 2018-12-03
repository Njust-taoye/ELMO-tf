import tensorflow as tf
from .char_cnn import CharCNNEmbedding


class ELMO:
    def __init__(self, config):
        self.embedding = CharCNNEmbedding(config)
        self.hidden_size = config["elmo_hidden"]
        self.vocab_size = config["word_vocab_size"]

        with tf.variable_scope("elmo_rnn", reuse=tf.AUTO_REUSE):
            self.forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, reuse=tf.AUTO_REUSE)
            self.backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, reuse=tf.AUTO_REUSE)

        with tf.variable_scope("elmo_project", reuse=tf.AUTO_REUSE):
            self.projection = tf.layers.Dense(self.vocab_size)

    def forward(self, data):
        embedding_output = self.embedding.forward(data)
        rnn_cell_outputs, state_outputs = tf.nn.bidirectional_dynamic_rnn(self.forward_cell, self.backward_cell,
                                                                          inputs=embedding_output,
                                                                          sequence_length=data["input_len"],
                                                                          dtype=tf.float32)
        # Concatenate the forward and backward LSTM output
        forward_backward_concat = tf.concat(rnn_cell_outputs, axis=-1)
        elmo_projection_output = self.projection(forward_backward_concat)
        return elmo_projection_output

    def train(self, data, global_step_variable=None):
        elmo_projection_output = self.forward(data)

        train_loss = tf.losses.sparse_softmax_cross_entropy(
            logits=elmo_projection_output,
            labels=data["target"]
        )
        train_ops = tf.train.AdamOptimizer().minimize(train_loss, global_step=global_step_variable)
        train_output = tf.nn.softmax(elmo_projection_output, dim=-1)
        return train_ops, train_loss, train_output

    def eval(self, data):
        elmo_projection_output = self.forward(data)
        eval_output = tf.nn.softmax(elmo_projection_output, dim=-1)
        return eval_output
