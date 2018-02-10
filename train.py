import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from toolkit import load_squad, load_glove

desc = 'Trainer for Machien QA'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--model', action='store',
                    help='What model to train?')
parser.add_argument('--logdir', action='store',
                    default='logs',
                    help='Where to store logs?')
parser.add_argument('--datadir', action='store',
                    default='data',
                    help='Where is the data stored?')
parser.add_argument('--list_models', action='store_true',
                    default=False,
                    help='List available models.')
args = parser.parse_args()

models = [i.replace('.py', '')
          for i in os.listdir("models")
          if '.py' in i and i[0] != '_']

if args.list_models:
    [print(m) for m in models]
    sys.exit(0)

# import the right model
config, build, feed_gen = None, None, None
try:
    ModelName = args.model
    if ModelName not in models:
        print("Please select a model which is available")
        [print(m) for m in models]
        sys.exit(0)
except IndexError:
    print('Available models')
import_string = 'from models.{} import build, feed_gen, config'
exec(import_string.format(ModelName))


if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)

ModelName = os.path.join(args.logdir, ModelName)

train_df = load_squad(os.path.join(args.datadir, 'train-v1.1.json'))
dev_df = load_squad(os.path.join(args.datadir, 'dev-v1.1.json'))
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
        out = sess.run(to_run, feed_dict=feed)
        losses.append(out[0])
    return np.array(losses)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(config.max_epochs):
        print(epoch, end=' ')
        output = run([out_dict['loss'], out_dict['trainer']],
                     config.train_steps, train_feed, inp_dict)
        train_writer.add_histogram("loss_dist", output, epoch)
        train_writer.add_scalar("loss", output.mean().reshape(-1), epoch)
        print('Train', output.mean(), end=' ')
        output = run([out_dict['loss']],
                     config.dev_steps, dev_feed, inp_dict)
        dev_writer.add_histogram("loss_dist", output, epoch)
        dev_writer.add_scalar("loss", output.mean().reshape(-1), epoch)
        saver.save(sess, ModelName, global_step=epoch)
        print('Dev', output.mean())
