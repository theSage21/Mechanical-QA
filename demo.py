import os
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from toolkit import load_squad, load_glove
from models.simple_rnn import build, feed_gen, config, ModelName


glove = load_glove('data/glove.6B.50d.txt')
config.build_trainer = False
inp_dict, out_dict = build(**config.__dict__)
