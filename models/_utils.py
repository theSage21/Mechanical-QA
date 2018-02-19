import numpy as np
import tensorflow as tf


__birnn = tf.nn.bidirectional_dynamic_rnn


def birnn(inputs, seq_lens, units, scope, cell_kind='gru'):
    cell = tf.contrib.rnn.LSTMCell
    if cell_kind.lower() == 'gru':
        cell = tf.contrib.rnn.GRUCell
    units_f = units // 2
    units_b = units - units_f
    with tf.variable_scope(scope):
        f = cell(units_f)
        b = cell(units_b)
        out, S = __birnn(cell_fw=f, cell_bw=b,
                         inputs=inputs, dtype=tf.float32,
                         scope=scope+'_birnn',
                         sequence_length=seq_lens)
    out = tf.concat(out, axis=2)
    return out, S


def dense(inp, hid_dim, scope):
    idim = inp.get_shape().as_list()[1]
    with tf.variable_scope(scope):
        w1 = tf.get_variable('w1', shape=(idim, hid_dim), dtype=tf.float32)
        b1 = tf.get_variable('b1', shape=(hid_dim, ), dtype=tf.float32)
        o = tf.matmul(inp, w1) + b1
    return o


def embed(sequence, table, default):
    return [table.get(i, default) for i in sequence]


def pad(sequence, length, pad):
    return sequence[:length] + [pad] * (length - len(sequence[:length]))


def make_glove(sequences, maxlen, glove, glovedim):
    gloves, lens = [], []
    for seq in sequences:
        gloves.append(embed(pad(seq, maxlen, '<<PADDING>>'),
                            glove, [0] * glovedim))
        lens.append(min(len(seq), maxlen))
    return np.array(gloves), np.array(lens)


def ohe(i, m):
    v = [0] * m
    if i < m:
        v[i] = 1
    return v

def gen_fractional_steps(lengths, max_lengths):
    fractions = [pad([1/l]*l, max_lengths, 0)
                 for l in lengths]
    return fractions
