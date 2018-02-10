import os
import sys
import json
import argparse
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from subprocess import check_output
from toolkit import load_squad, load_glove, get_answer

desc = 'Evaluator for Machine QA'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--prediction', action='store',
                    default='predict.json',
                    help='Where to put the predictions of the model?')
parser.add_argument('--checkpoint', action='store',
                    help='Path to checkpoint that has to be loaded')
parser.add_argument('--model', action='store',
                    help='What model to evaluate?')
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
config.build_trainer = False
config.batch_size = 1

# --------------evaluation
dev_path = os.path.join(args.datadir, 'dev-v1.1.json')
dev_df = load_squad(dev_path)
print(dev_df.shape[0], 'dev size')
glove = load_glove('data/glove.6B.50d.txt')
inp_dict, out_dict = build(**config.__dict__)
dev_feed = feed_gen(dev_df, glove=glove, infinite=False, **config.__dict__)


answers = {}
saver = tf.train.Saver()
with tf.Session() as sess:
    print('Restoring ', args.checkpoint)
    saver.restore(sess, args.checkpoint)
    for batch in tqdm(dev_feed, total=dev_df.shape[0], desc='Predicting'):
        feed = {v: batch[k] for k, v in inp_dict.items()}
        out = sess.run([out_dict['start_pred'],
                        out_dict['end_pred']],
                       feed_dict=feed)
        ans = get_answer(out[0], out[1], batch)
        answers.update(ans)
with open(args.prediction, 'w') as fl:
    json.dump(answers, fl)
command = "python data/evalsquad.py {data} {pred}"
command = command.format(data=dev_path, pred=args.prediction)
output = check_output(command, shell=True).decode()
results = eval(output.strip().split('\n')[-1])
table = '{dt:30} | {model:<20} | {f1:<10} | {em:<10}'
row = table.format(dt=str(datetime.now()),
                   model=args.checkpoint,
                   f1=round(results['f1'], 7),
                   em=round(results['exact_match'], 7))
with open('README.md', 'a') as fl:
    fl.write(row+'\n')
