#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TRNNConfig(object):
    """RNN配置參數"""

    # 模型參數
    embedding_dim = 64      # 詞向量維度
    seq_length = 600        # 序列長度
    num_classes = 10        # 類別數
    vocab_size = 5000       # 詞彙表大小

    num_layers = 2           # 隱藏層層數
    hidden_dim = 128        # 隱藏層神經元
    rnn = 'lstm'             # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3    # 學習率

    batch_size = 128         # 每批訓練大小
    num_epochs = 10          # 總反覆運算輪次

    print_per_batch = 100    # 每多少輪輸出一次結果
    save_per_batch = 10      # 每多少輪存入tensorboard


class TextRNN(object):
    """文本分類，RNN模型"""

    def __init__(self, config):
        self.config = config

        # 三個待輸入的資料
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 為每一個rnn核後面加一個dropout層
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 詞向量映射
        with tf.device('/gpu:0'):
            embedding = tf.get_variable(
                'embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多層rnn網路
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最後一個時序輸出作為結果

        with tf.name_scope("score"):
            # 全連接層，後面接dropout以及relu啟動
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分類器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 預測類別

        with tf.name_scope("optimize"):
            # 損失函數，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 優化器
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 準確率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
