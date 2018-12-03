import tensorflow as tf

from trainer import ELMOTrainer
from config import config_dict

trainer = ELMOTrainer(config_dict)
trainer.train()