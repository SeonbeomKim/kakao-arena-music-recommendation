# https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
# https://arxiv.org/abs/1607.06450 Layer Normalization
# https://arxiv.org/abs/1512.00567 Label Smoothing

import tensorflow as tf  # version 1.4


# tf.set_random_seed(787)

class Transformer:
    def __init__(self, voca_size, embedding_size, is_embedding_scale, max_sequence_length,
                 encoder_decoder_stack, multihead_num, pad_idx, unk_idx):

        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.is_embedding_scale = is_embedding_scale  # True or False
        self.max_sequence_length = max_sequence_length
        self.encoder_decoder_stack = encoder_decoder_stack
        self.multihead_num = multihead_num
        self.pad_idx = pad_idx  # <'pad'> symbol index
        self.unk_idx = unk_idx  # <'pad'> symbol index

        with tf.name_scope("placeholder"):
            self.lr = tf.placeholder(tf.float32)

            # cls_idx || song indices || sep_idx || tag indices || sep_idx 로 받음. (max_sequence_length) 길이
            self.input_sequence_indices = tf.placeholder(tf.int32, [None, None], name='input_sequence_indices')

            # cls_idx || song indices || sep_idx 까지가 A 길이
            self.A_length = tf.placeholder(tf.int32, [None], name='A_length')
            self.next_item_idx = tf.placeholder(tf.int32, [None, None], name='next_item_idx')
            self.negative_item_idx = tf.placeholder(tf.int32, [None, None], name='negative_item_idx')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE) and (attention)

            # pre_training 용도
            self.boolean_mask = tf.placeholder(tf.bool, [None, None], name='boolean_mask')
            self.masked_LM_target = tf.placeholder(tf.int32, [None], name='masked_LM_target')
            self.label_smoothing = tf.placeholder(tf.float32, name='label_smoothing')

        with tf.name_scope("embedding_table"):
            self.embedding_table = tf.get_variable(
                # https://github.com/tensorflow/models/blob/master/official/transformer/model/embedding_layer.py
                'embedding_table',
                [self.voca_size, self.embedding_size],
                initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5))

            # https://github.com/google-research/bert/blob/master/modeling.py
            self.position_embedding_table = tf.get_variable(  # [self.max_sequence_length, self.embedding_size]
                'position_embedding_table',
                [self.max_sequence_length, self.embedding_size],  # (cls + sep + eos) 총 3개 추
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            self.segment_embedding_table = tf.get_variable(  # [2, self.embedding_size]
                'segment_embedding_table',
                [2, self.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.name_scope('encoder'):
            encoder_input_embedding, encoder_input_mask = self.embedding_and_PE(self.input_sequence_indices,
                                                                                self.A_length)
            self.encoder_embedding = self.encoder(encoder_input_embedding, encoder_input_mask)

        with tf.name_scope('sequence_embedding'):
            # get cls embedding
            self.sequence_embedding = self.encoder_embedding[:, 0, :]  # [N, self.embedding_size]

        with tf.name_scope('train_sequence_embedding'):
            next_item_embedding = tf.nn.embedding_lookup(
                self.embedding_table,
                self.next_item_idx)  # [N, K, self.embedding_size]
            negative_item_embedding = tf.nn.embedding_lookup(
                self.embedding_table,
                self.negative_item_idx)  # [N, K, self.embedding_size]

            positive = tf.reduce_sum(  # [N, K]
                tf.expand_dims(self.sequence_embedding, axis=1) * next_item_embedding,  # [N, K, self.embedding_size]
                axis=-1)
            negative = tf.reduce_sum(  # [N, K]
                tf.expand_dims(self.sequence_embedding, axis=1) * negative_item_embedding,
                # [N, K, self.embedding_size]
                axis=-1)

            # next item에 대해서는 sigmoid 결과 1, negative item에 대해서는 sigmoid 결과 0 되도록 학습
            positive_loss = tf.nn.sigmoid_cross_entropy_with_logits(  # [N, K]
                labels=tf.ones_like(positive),
                logits=positive)
            # unk인 label 위치는 학습안되도록 masking
            positive_loss_mask = tf.cast(tf.not_equal(self.next_item_idx, self.unk_idx), tf.float32)  # [N, K]
            valid_positive_size = tf.reduce_sum(positive_loss_mask, axis=-1)  # [N]
            self.positive_loss = tf.reduce_sum( # 상수
                tf.reduce_sum(positive_loss * positive_loss_mask, # [N]
                              axis=-1) / valid_positive_size)

            negative_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(negative),
                logits=negative)
            # unk인 label 위치는 학습안되도록 masking
            negative_loss_mask = tf.cast(tf.not_equal(self.negative_item_idx, self.unk_idx), tf.float32)  # [N, K]
            valid_negative_size = tf.reduce_sum(negative_loss_mask, axis=-1)  # [N]
            self.negative_loss = tf.reduce_sum( # 상수
                tf.reduce_sum(negative_loss * negative_loss_mask, # [N]
                              axis=-1) / valid_negative_size)

            # 학습할 objective function
            self.loss = self.positive_loss + self.negative_loss

        with tf.name_scope('masked_pre_training'):
            self.masked_position = tf.boolean_mask(self.encoder_embedding,
                                                   self.boolean_mask)  # [np.sum(boolean_mask), self.embedding_size]
            self.masked_LM_pred = tf.matmul(self.masked_position, tf.transpose(
                self.embedding_table))  # [np.sum(boolean_mask), self.voca_size]

            # make smoothing target one hot vector
            self.masked_LM_target_one_hot = tf.one_hot(
                self.masked_LM_target,
                depth=self.voca_size,
                on_value=(1.0 - self.label_smoothing) + (self.label_smoothing / self.voca_size),  # tf.float32
                off_value=(self.label_smoothing / self.voca_size),  # tf.float32
                dtype=tf.float32
            )  # [np.sum(boolean_mask), self.voca_size]

            self.masked_LM_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.masked_LM_target_one_hot,
                logits=self.masked_LM_pred))  # [np.sum(boolean_mask)]

        with tf.name_scope('train_metric'):
            self.masked_LM_pred_argmax = tf.argmax(self.masked_LM_pred, 1,
                                                   output_type=tf.int32)  # [np.sum(boolean_mask)]
            self.masked_LM_correct = tf.reduce_sum(
                tf.cast(tf.equal(self.masked_LM_pred_argmax, self.masked_LM_target), tf.int32)
            )

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
            self.minimize = optimizer.minimize(self.loss)
            self.masked_LM_minimize = optimizer.minimize(self.masked_LM_loss)

        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(max_to_keep=10000)

    def embedding_and_PE(self, data, A_length):
        # data: [N, data_length]

        # embedding lookup and scale
        embedding = tf.nn.embedding_lookup(  # [N, data_length, self.embedding_size]
            self.embedding_table,
            data)

        A_B_sequence_mask = tf.sequence_mask(
            A_length,
            tf.shape(data)[1])  # [N, data_length], A:1, B:0

        # tag 부분은 PE를 더하면 안돼서 마스킹
        PE_mask = tf.expand_dims(tf.cast(A_B_sequence_mask, tf.float32), axis=-1)  # [N, data_length, 1]
        PE = tf.expand_dims(  # [1, data_length, self.embedding_size], will be broadcast
            self.position_embedding_table,
            axis=0)[:, :tf.shape(data)[1], :]
        PE = PE * PE_mask

        SE = tf.nn.embedding_lookup(  # [N, data_length, self.embedding_size]
            self.segment_embedding_table,
            tf.cast(A_B_sequence_mask, tf.int32))

        if self.is_embedding_scale is True:
            embedding *= self.embedding_size ** 0.5

        embedding_mask = tf.expand_dims(
            tf.cast(tf.not_equal(data, self.pad_idx), dtype=tf.float32),  # [N, data_length]
            axis=-1
        )  # [N, data_length, 1]

        # Add Position Encoding
        embedding += (PE + SE)

        # pad masking (set 0 PE added pad position)
        embedding *= embedding_mask

        # Layer Normalization
        embedding = tf.contrib.layers.layer_norm(embedding, begin_norm_axis=2)

        # Drop out
        embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)
        return embedding, embedding_mask

    def encoder(self, encoder_input_embedding, encoder_input_mask):
        # encoder_input_embedding: [N, self.encoder_input_length, self.embedding_size] , pad mask applied
        # encoder_input_mask: [N, self.encoder_input_length, 1]

        # mask
        encoder_self_attention_mask = tf.tile(
            tf.matmul(encoder_input_mask, tf.transpose(encoder_input_mask, [0, 2, 1])),
            # [N, encoder_input_length, encoder_input_length]
            [self.multihead_num, 1, 1]
        )  # [self.multihead_num*N, encoder_input_length, encoder_input_length]

        for i in range(self.encoder_decoder_stack):  # 6
            # Multi-Head Attention
            Multihead_add_norm = self.multi_head_attention_add_norm(
                query=encoder_input_embedding,
                key_value=encoder_input_embedding,
                score_mask=encoder_self_attention_mask,
                output_mask=encoder_input_mask,
                activation=None,
                name='encoder' + str(i)
            )  # [N, self.encoder_input_length, self.embedding_size]

            # Feed Forward
            encoder_input_embedding = self.dense_add_norm(
                Multihead_add_norm,
                self.embedding_size,
                output_mask=encoder_input_mask,  # set 0 bias added pad position
                activation=tf.nn.relu,
                name='encoder_dense' + str(i)
            )  # [N, self.encoder_input_length, self.embedding_size]

        return encoder_input_embedding  # [N, self.encoder_input_length, self.embedding_size]

    def multi_head_attention_add_norm(self, query, key_value, score_mask=None, output_mask=None, activation=None,
                                      name=None):
        # Sharing Variables
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # for문으로 self.multihead_num번 돌릴 필요 없이 embedding_size 만큼 만들고 self.multihead_num등분해서 연산하면 됨.
            V = tf.layers.dense(  # layers dense는 배치(N)별로 동일하게 연산됨.
                key_value,
                units=self.embedding_size,
                activation=activation,
                use_bias=False,
                name='V'
            )  # [N, key_value_sequence_length, self.embedding_size]
            K = tf.layers.dense(
                key_value,
                units=self.embedding_size,
                activation=activation,
                use_bias=False,
                name='K'
            )  # [N, key_value_sequence_length, self.embedding_size]
            Q = tf.layers.dense(
                query,
                units=self.embedding_size,
                activation=activation,
                use_bias=False,
                name='Q'
            )  # [N, query_sequence_length, self.embedding_size]

            # linear 결과를 self.multihead_num등분하고 연산에 지장을 주지 않도록 batch화 시킴.
            # https://github.com/Kyubyong/transformer 참고.
            # split: [N, key_value_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재
            V = tf.concat(tf.split(V, self.multihead_num, axis=-1),
                          axis=0)  # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
            K = tf.concat(tf.split(K, self.multihead_num, axis=-1),
                          axis=0)  # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
            Q = tf.concat(tf.split(Q, self.multihead_num, axis=-1),
                          axis=0)  # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]

            # Q * (K.T) and scaling ,  [self.multihead_num*N, query_sequence_length, key_value_sequence_length]
            score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size / self.multihead_num)

            # masking
            if score_mask is not None:
                score *= score_mask  # zero mask
                score += ((score_mask - 1) * 1e+9)  # -inf mask
            # decoder self_attention:
            # 1 0 0
            # 1 1 0
            # 1 1 1 형태로 마스킹

            # encoder_self_attention
            # if encoder_input_data: i like </pad>
            # 1 1 0
            # 1 1 0
            # 0 0 0 형태로 마스킹

            # ED_attention
            # if encoder_input_data: i like </pad>
            # 1 1 0
            # 1 1 0
            # 1 1 0 형태로 마스킹

            softmax = tf.nn.softmax(score,
                                    dim=2)  # [self.multihead_num*N, query_sequence_length, key_value_sequence_length]

            # Attention dropout
            # https://arxiv.org/abs/1706.03762v4 => v4 paper에는 attention dropout 하라고 되어 있음.
            softmax = tf.nn.dropout(softmax, keep_prob=self.keep_prob)

            # Attention weighted sum
            attention = tf.matmul(softmax,
                                  V)  # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]

            # split: [N, query_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재
            concat = tf.concat(tf.split(attention, self.multihead_num, axis=0),
                               axis=-1)  # [N, query_sequence_length, self.embedding_size]

            # Linear
            Multihead = tf.layers.dense(
                concat,
                units=self.embedding_size,
                activation=activation,
                use_bias=False,
                name='linear'
            )  # [N, query_sequence_length, self.embedding_size]

            if output_mask is not None:
                Multihead *= output_mask

            # residual Drop Out
            Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
            # Add
            Multihead += query
            # Layer Norm
            Multihead = tf.contrib.layers.layer_norm(Multihead,
                                                     begin_norm_axis=2)  # [N, query_sequence_length, self.embedding_size]

            return Multihead

    def dense_add_norm(self, embedding, units, output_mask=None, activation=None, name=None):
        # FFN(x) = max(0, x*W1+b1)*W2 + b2
        # Sharing Variables
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            inner_layer = tf.layers.dense(
                embedding,
                units=4 * self.embedding_size,  # bert paper
                activation=activation  # relu
            )  # [N, self.decoder_input_length, 4*self.embedding_size]
            dense = tf.layers.dense(
                inner_layer,
                units=units,
                activation=None
            )  # [N, self.decoder_input_length, self.embedding_size]

            if output_mask is not None:
                dense *= output_mask  # set 0 bias added pad position

            # Drop out
            dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
            # Add
            dense += embedding
            # Layer Norm
            dense = tf.contrib.layers.layer_norm(dense, begin_norm_axis=2)

        return dense
