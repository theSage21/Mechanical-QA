import os
import numpy as np
import tensorflow as tf
from tqdm import trange
from tensorboardX import SummaryWriter
from toolkit import load_squad, load_glove
from models.simple_rnn import build, feed_gen, config


if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

ModelName = 'checkpoints/SimpleRNN'

train_df = load_squad('data/train-v1.1.json')
dev_df = load_squad('data/dev-v1.1.json')
train_df = train_df.loc[[i < config.max_c_len
                         for i in train_df['start_exp_one_hot']]]
dev_df = dev_df.loc[[i < config.max_c_len
                     for i in dev_df['start_exp_one_hot']]]
glove = load_glove('data/glove.6B.50d.txt')

inp_dict, out_dict = build(**config.__dict__)
train_feed = feed_gen(train_df, glove=glove, **config.__dict__)
dev_feed = feed_gen(dev_df, glove=glove, **config.__dict__)


train_writer = SummaryWriter('logs/train')
dev_writer = SummaryWriter('logs/dev')


def run(to_run, steps, feeder, input_dict):
    losses = []
    for _, b in zip(range(steps), feeder):
        feed = {v: b[k] for k, v in inp_dict.items()}
        l = sess.run(to_run, feed_dict=feed)
        losses.append(l[0])
    return np.array(losses)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in trange(config.max_epochs):
        l = run([out_dict['loss'], out_dict['trainer']],
                config.train_steps, train_feed, inp_dict)
        train_writer.add_histogram("loss_dist", l, epoch)
        train_writer.add_scalar("loss", l.mean().reshape(-1), epoch)
        l = run([out_dict['loss']],
                config.dev_steps, dev_feed, inp_dict)
        dev_writer.add_histogram("loss_dist", l, epoch)
        dev_writer.add_scalar("loss", l.mean().reshape(-1), epoch)
        saver.save(sess, ModelName, global_step=epoch)
