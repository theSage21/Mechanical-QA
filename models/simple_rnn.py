import tensorflow as tf
from .utils import birnn, dense, make_glove


def build_simple_rnn(*, batch_size, max_c_len, max_q_len,
                     glove_dim, summary_dim, reasoning_dim,
                     build_trainer=True):
    with tf.variable_scope("placeholders"):
        c_glove = tf.placeholder(name="c_glove",
                                 shape=(batch_size,
                                        max_c_len,
                                        glove_dim),
                                 dtype=tf.float32)
        q_glove = tf.placeholder(name="q_glove",
                                 shape=(batch_size,
                                        max_q_len,
                                        glove_dim),
                                 dtype=tf.float32)
        start_exp = tf.placeholder(name="start_exp",
                                   shape=(batch_size,
                                          max_c_len),
                                   dtype=tf.float32)
        end_exp = tf.placeholder(name="end_exp",
                                 shape=(batch_size,
                                        max_c_len),
                                 dtype=tf.float32)
        c_len = tf.placeholder(name='c_len',
                               shape=(batch_size,),
                               dtype=tf.int32)
        q_len = tf.placeholder(name='q_len',
                               shape=(batch_size,),
                               dtype=tf.int32)
    with tf.variable_scope("summaries"):
        _, c_s = birnn(c_glove, c_len, summary_dim, 'context_summary')
        _, q_s = birnn(q_glove, q_len, summary_dim, 'question_summary')
        summaries = tf.concat([*c_s, *q_s], axis=-1)
    with tf.variable_scope("reasoning"):
        understand = summaries
        for i in range(3):
            understand = tf.nn.tanh(dense(understand, reasoning_dim,
                                          'reason_'+str(i)))
    with tf.variable_scope("boundary_prediction"):
        start_pred = tf.nn.relu(dense(understand, max_c_len, 'start'))
        end_pred = tf.nn.relu(dense(understand, max_c_len, 'end'))
        # masks for pointers
        pm = 1 - tf.cumsum(tf.one_hot(c_len, max_c_len), axis=1)
        start_pred = tf.multiply(start_pred, pm)
        end_pred = tf.multiply(end_pred, pm)
    inp_dict = {"c_glove": c_glove, "q_glove": q_glove,
                "start_exp": start_exp, "end_exp": end_exp,
                "c_len": c_len, "q_len": q_len}
    out_dict = {"start_pred": start_pred, "end_pred": end_pred}
    if build_trainer:
        with tf.variable_scope("trainer"):
            sl = tf.nn.softmax_cross_entropy_with_logits(logits=start_pred,
                                                         labels=start_exp)
            el = tf.nn.softmax_cross_entropy_with_logits(logits=end_pred,
                                                         labels=end_exp)
            loss = tf.reduce_mean(sl + el)
            train = tf.train.AdamOptimizer().minimize(loss)
        out_dict['loss'] = loss
        out_dict['trainer'] = train
    return inp_dict, out_dict


def data_feed(dataset, batch_size, glove,
              max_c_len, max_q_len,
              glove_dim):
    while True:
        df = dataset.sample(dataset.shape[0]).copy()
        for i in range(0, df.shape[0]//batch_size, batch_size):
            part = df[i: i + batch_size]
            c_glove, c_len = make_glove(part['c_tokens'],
                                        max_c_len,
                                        glove, glove_dim)
            q_glove, q_len = make_glove(part['q_tokens'],
                                        max_q_len,
                                        glove, glove_dim)
            start_exp = part['start_exp_one_hot']
            end_exp = part['end_exp_one_hot']
            feed = {"c_glove": c_glove, "q_glove": q_glove,
                    "start_exp": start_exp, "end_exp": end_exp,
                    "c_len": c_len, "q_len": q_len}
            yield feed
