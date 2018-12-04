import tensorflow as tf

from trainer import ELMOTrainer
from config import config_dict

trainer = ELMOTrainer(config_dict)
trainer.train()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(config_dict["log_dir"] + '/train',
                                     trainer.sess.graph,
                                     filename_suffix=config_dict["log_file_prefix"])
