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
