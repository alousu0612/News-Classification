# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置參數"""

    embedding_dim = 64  # 詞向量維度
    seq_length = 600  # 序列長度
    num_classes = 10  # 類別數
    num_filters = 256  # 卷積核數目
    kernel_size = 5  # 卷積核尺寸
    vocab_size = 5000  # 詞彙表達小

    hidden_dim = 128  # 全連接層神經元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 學習率

    batch_size = 64  # 每批訓練大小
    num_epochs = 10  # 總反覆運算輪次

    print_per_batch = 100  # 每多少輪輸出一次結果
    save_per_batch = 10  # 每多少輪存入tensorboard


class TextCNN(object):
    """文本分類，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三個待輸入的資料
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 詞向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全連接層，後面接dropout以及relu啟動
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分類器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 預測類別

        with tf.name_scope("optimize"):
            # 損失函數，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 優化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 準確率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
