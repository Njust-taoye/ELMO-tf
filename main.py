import tensorflow as tf

from trainer import ELMOTrainer
from config import config_dict

trainer = ELMOTrainer(config_dict)
trainer.train()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(config_dict["log_dir"] + '/train', sess.graph)
test_writer = tf.summary.FileWriter(config_dict["log_dir"] + '/test')
