import numpy as np
import pandas as pd
import tensorflow as tf
from ._utils import birnn, dense, make_glove, gen_fractional_steps


def build(*, batch_size, max_c_len, max_q_len,
          glove_dim, summary_dim, reasoning_dim,
          keep_proba,
          build_trainer=True, **other_kwargs):
    print('Building BRAHMA model')
    with tf.variable_scope("placeholders"):
        c_glove = tf.placeholder(name="c_glove",
                                 shape=(batch_size,
                                        max_c_len,
                                        glove_dim),
                                 dtype=tf.float32)
        c_steps = tf.placeholder(name='c_steps',
                                 shape=(batch_size,
                                        max_c_len,
                                        1),
                                 dtype=tf.float32)
        q_glove = tf.placeholder(name="q_glove",
                                 shape=(batch_size,
                                        max_q_len,
                                        glove_dim),
                                 dtype=tf.float32)
        start_exp = tf.placeholder(name="start_exp",
                                   shape=(batch_size,
                                          1),
                                   dtype=tf.float32)
        end_exp = tf.placeholder(name="end_exp",
                                 shape=(batch_size,
                                        1),
                                 dtype=tf.float32)
        c_len = tf.placeholder(name='c_len',
                               shape=(batch_size,),
                               dtype=tf.int32)
        q_len = tf.placeholder(name='q_len',
                               shape=(batch_size,),
                               dtype=tf.int32)
    with tf.variable_scope("summaries"):
        inp = q_glove
        for x in range(1):
            inp, q_s = birnn(inp, q_len, summary_dim,
                             'question_summary'+str(x))
        summaries = tf.concat(q_s, axis=-1)
        summaries = tf.nn.dropout(summaries, keep_proba)
        summaries = tf.tile(tf.expand_dims(summaries, axis=1),
                            [1, max_c_len, 1])
    with tf.variable_scope("reasoning"):
        rep = tf.concat([c_glove, summaries, c_steps], axis=-1)
        for x in range(1):
            rep, c_s = birnn(rep, c_len, summary_dim,
                             'context_reasoning'+str(x))
        understand = tf.concat(c_s, axis=-1)
    with tf.variable_scope("boundary_prediction"):
        start_pred = tf.nn.tanh(dense(understand, reasoning_dim, 'start1'))
        start_pred = tf.nn.tanh(dense(start_pred, reasoning_dim, 'start2'))
        start_pred = tf.nn.sigmoid(dense(start_pred, 1, 'start'))
        end_pred = tf.nn.tanh(dense(understand, reasoning_dim, 'end1'))
        end_pred = tf.nn.tanh(dense(end_pred, reasoning_dim, 'end2'))
        end_pred = tf.nn.sigmoid(dense(end_pred, 1, 'end'))
        # masks for pointers
        start_index = tf.floor(start_pred * tf.cast(tf.expand_dims(c_len, axis=-1), tf.float32))
        end_index = tf.floor(end_pred * tf.cast(tf.expand_dims(c_len, axis=-1), tf.float32))
    inp_dict = {"c_glove": c_glove, "q_glove": q_glove,
                "start_exp": start_exp, "end_exp": end_exp,
                'c_steps': c_steps,
                "c_len": c_len, "q_len": q_len}
    out_dict = {"start_pred": start_pred, "end_pred": end_pred,
                "start_index": start_index, "end_index": end_index}
    if build_trainer:
        with tf.variable_scope("trainer"):
            sl = tf.square((start_exp - start_pred))
            el = tf.square((end_exp - end_pred))
            loss = tf.reduce_mean(sl + el)
            train = tf.train.AdamOptimizer().minimize(loss)
        out_dict['loss'] = loss
        out_dict['trainer'] = train
    return inp_dict, out_dict


def feed_gen(dataset, *, batch_size, glove,
             max_c_len, max_q_len,
             glove_dim, infinite=True, **other_kwargs):
    while True:
        df = dataset.sample(dataset.shape[0]).copy()
        for i in range(0, df.shape[0], batch_size):
            part = df[i: i + batch_size].copy()
            while part.shape[0] != batch_size:
                part = pd.concat([part, part])[:batch_size]
            c_glove, c_len = make_glove(part['c_tokens'],
                                        max_c_len,
                                        glove, glove_dim)
            c_steps = gen_fractional_steps(c_len, max_c_len)
            q_glove, q_len = make_glove(part['q_tokens'],
                                        max_q_len,
                                        glove, glove_dim)
            start_exp = [[i/l] for i, l in zip(part['start'], c_len)]
            end_exp = [[i/l] for i, l in zip(part['end'], c_len)]
            feed = {"c_glove": c_glove, "q_glove": q_glove,
                    "start_exp": start_exp, "end_exp": end_exp,
                    "c_len": c_len, "q_len": q_len,
                    'c_steps': c_steps,
                    'qid': part['qid'], 'c_tokens': part['c_tokens']}
            feed = {k: np.array(v) for k, v in feed.items()}
            yield feed
        if not infinite:
            break


class Config:
    def __init__(self):
        self.max_epochs = 5000
        self.train_steps = 50
        self.dev_steps = 50
        self.batch_size = 128
        self.max_c_len = 400
        self.max_q_len = 30
        self.glove_dim = 50
        self.summary_dim = 128
        self.reasoning_dim = 128
        self.build_trainer = True
        self.keep_proba = 0.8


config = Config()
if __name__ == '__main__':
    build(**config.__dict__)
