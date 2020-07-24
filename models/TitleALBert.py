import tensorflow as tf  # version 1.4
import numpy as np

class TitleALBert:
    def __init__(self, voca_size, embedding_size, hidden_size, is_embedding_scale, max_sequence_length,
                 encoder_decoder_stack, multihead_num, pad_idx, songs_num, tags_num):
        tf.set_random_seed(787)

        self.voca_size = voca_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.is_embedding_scale = is_embedding_scale  # True or False
        self.max_sequence_length = max_sequence_length
        self.encoder_decoder_stack = encoder_decoder_stack
        self.multihead_num = multihead_num
        self.pad_idx = pad_idx  # <'pad'> symbol index
        self.songs_num = songs_num  # song
        self.tags_num = tags_num

        with tf.name_scope("placeholder"):
            self.lr = tf.placeholder(tf.float32)
            self.tags_loss_weight = tf.placeholder(tf.float32)
            self.song_top_k = tf.placeholder(tf.int32)
            self.tag_top_k = tf.placeholder(tf.int32)

            self.input_sequence_indices = tf.placeholder(tf.int32, [None, None], name='input_sequence_indices')
            self.sparse_label = tf.placeholder(dtype=tf.int64, shape=[None, 2])
            self.batch_size = tf.placeholder(dtype=tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # pretrain
            self.boolean_mask = tf.placeholder(tf.bool, [None, None], name='boolean_mask')
            self.masked_LM_label = tf.placeholder(tf.int32, [None], name='masked_LM_target')

        with tf.name_scope("embedding_table"):
            self.embedding_table = tf.get_variable(
                'embedding_table',
                [self.voca_size, self.embedding_size],
                initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5))

            songs_tags_embedding_table = tf.get_variable(
                "songs_tags_embedding_table",
                [self.songs_num + self.tags_num, self.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.songs_tags_embedding_table = tf.nn.dropout(songs_tags_embedding_table,
                                                            keep_prob=self.keep_prob)

            self.position_embedding_table = tf.get_variable(
                'position_embedding_table',
                [self.max_sequence_length, self.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.name_scope('encoder'):
            encoder_input_embedding, encoder_input_mask = self.embedding_and_PE(self.input_sequence_indices)
            self.encoder_embedding = self.encoder(encoder_input_embedding, encoder_input_mask)

        with tf.name_scope('sequence_embedding'):
            # get cls embedding
            self.song_cls_embedding = self.encoder_embedding[:, 0, :]  # [N, self.embedding_size]
            self.tag_cls_embedding = self.encoder_embedding[:, 1, :]  # [N, self.embedding_size]

        with tf.name_scope('label'):
            label_sparse_tensor = tf.SparseTensor(indices=self.sparse_label,
                                                  values=tf.ones(tf.shape(self.sparse_label)[0]),
                                                  dense_shape=[self.batch_size,
                                                               self.songs_num + self.tags_num])
            label = tf.sparse_tensor_to_dense(label_sparse_tensor, validate_indices=False)

            song_label = label[:, :self.songs_num]
            tag_label = label[:, self.songs_num:self.songs_num + self.tags_num]

        with tf.name_scope('trainer'):
            song_embedding_table = self.songs_tags_embedding_table[:self.songs_num, :]
            tag_embedding_table = self.songs_tags_embedding_table[
                                  self.songs_num:self.songs_num + self.tags_num, :]

            self.songs_bias = tf.get_variable('songs_bias', [1, self.songs_num], initializer=tf.zeros_initializer())
            self.tags_bias = tf.get_variable('tags_bias', [1, self.tags_num], initializer=tf.zeros_initializer())

            song_predict = tf.matmul(self.song_cls_embedding, song_embedding_table, transpose_b=True) + self.songs_bias
            tag_predict = tf.matmul(self.tag_cls_embedding, tag_embedding_table, transpose_b=True) + self.tags_bias

            song_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=song_label, logits=song_predict)
            tag_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tag_label, logits=tag_predict)

        with tf.name_scope('ranking_loss'):
            sigmoid_song_predict = tf.nn.sigmoid(song_predict)

            # index별 자기 자신의 ranking 담김 ex) song_predict = [3,1,2,5] -> ranking = [1, 3, 2, 0]
            song_predict_ranking = tf.argsort(tf.argsort(sigmoid_song_predict, axis=-1, direction='DESCENDING'),
                                              axis=-1,
                                              direction='ASCENDING')

            top_k = 100  # 1000
            top_k_song_predict_label = tf.cast(song_predict_ranking < top_k, dtype=tf.float32)
            # top_k 중 진짜 정답 위치
            correct_ranking_label = song_label * top_k_song_predict_label  # [N, song_num]

            wrong_top_k = 100  # 1000
            wrong_top_k_song_predict_label = tf.cast(song_predict_ranking < wrong_top_k, dtype=tf.float32)
            # top_k 중 틀린 정답 위치
            wrong_ranking_label = (1 - song_label) * wrong_top_k_song_predict_label

            song_ranking_loss = -(correct_ranking_label * tf.log(sigmoid_song_predict + 1e-10) + (
                    wrong_ranking_label * tf.log(1 - sigmoid_song_predict + 1e-10)))

        with tf.name_scope('total_loss'):
            self.loss = tf.reduce_mean(song_loss) + self.tags_loss_weight * tf.reduce_mean(tag_loss)
            self.loss_with_ranking_loss = self.loss + 2 * tf.reduce_mean(song_ranking_loss)

        with tf.name_scope('predictor'):
            self.reco_songs, self.reco_songs_score = self.top_k(
                predict=tf.nn.sigmoid(song_predict),
                top_k=self.song_top_k)
            self.reco_tags, self.reco_tags_score = self.top_k(
                predict=tf.nn.sigmoid(tag_predict),
                top_k=self.tag_top_k)
            self.reco_tags += self.songs_num

        with tf.name_scope('pretrain'):
            self.masked_position = tf.boolean_mask(self.encoder_embedding,
                                                   self.boolean_mask)  # [np.sum(boolean_mask), self.embedding_size]
            self.masked_LM_pred = tf.matmul(self.masked_position, tf.transpose(
                self.embedding_table))  # [np.sum(boolean_mask), self.voca_size]

            self.label_smoothing = 0.1
            self.masked_LM_label_one_hot = tf.one_hot(
                self.masked_LM_label,
                depth=self.voca_size,
                on_value=(1.0 - self.label_smoothing) + (self.label_smoothing / self.voca_size),  # tf.float32
                off_value=(self.label_smoothing / self.voca_size),  # tf.float32
                dtype=tf.float32
            )  # [np.sum(boolean_mask), self.voca_size]

            self.masked_LM_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.masked_LM_label_one_hot,
                logits=self.masked_LM_pred
            ))  # [np.sum(boolean_mask)]

            self.masked_LM_pred_argmax = tf.argmax(self.masked_LM_pred, 1,
                                                   output_type=tf.int32)  # [np.sum(boolean_mask)]
            self.masked_LM_correct = tf.reduce_sum(
                tf.cast(tf.equal(self.masked_LM_pred_argmax, self.masked_LM_label), tf.int32)
            )

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
            self.minimize = optimizer.minimize(self.loss)
            self.minimize_masked_LM_loss = optimizer.minimize(self.masked_LM_loss)
            self.minimize_with_ranking_loss = optimizer.minimize(self.loss_with_ranking_loss)

        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(max_to_keep=10000)


    def top_k(self, predict, top_k=100):
        reco = tf.math.top_k(  # [N, label]
            predict,
            top_k)

        return reco.indices, reco.values  # [N, top_k]

    def embedding_and_PE(self, data):
        # data: [N, data_length]

        # embedding lookup and scale
        embedding = tf.nn.embedding_lookup(  # [N, data_length, self.embedding_size]
            self.embedding_table,
            data)
        embedding = tf.layers.dense(embedding, units=self.hidden_size, activation=None, use_bias=False)

        PE = tf.expand_dims(  # [1, data_length, self.embedding_size], will be broadcast
            self.position_embedding_table,
            axis=0)[:, :tf.shape(data)[1], :]

        if self.is_embedding_scale is True:
            embedding *= self.hidden_size ** 0.5

        embedding_mask = tf.expand_dims(
            tf.cast(tf.not_equal(data, self.pad_idx), dtype=tf.float32),  # [N, data_length]
            axis=-1
        )  # [N, data_length, 1]

        # Add Position Encoding
        embedding += PE

        # pad masking (set 0 PE added pad position)
        embedding *= embedding_mask

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
            # PRENORM
            encoder_input_embedding = tf.contrib.layers.layer_norm(encoder_input_embedding, begin_norm_axis=2)

            # Multi-Head Attention
            Multihead_add_norm = self.multi_head_attention_add_norm(
                query=encoder_input_embedding,
                key_value=encoder_input_embedding,
                score_mask=encoder_self_attention_mask,
                output_mask=encoder_input_mask,
                activation=None,
                name='encoder'
            )  # [N, self.encoder_input_length, self.embedding_size]

            # PRENORM
            Multihead_add_norm = tf.contrib.layers.layer_norm(Multihead_add_norm, begin_norm_axis=2)

            # Feed Forward
            encoder_input_embedding = self.dense_add_norm(
                Multihead_add_norm,
                self.hidden_size,
                output_mask=encoder_input_mask,  # set 0 bias added pad position
                activation=self.gelu,
                name='encoder_dense' + str(i)
            )  # [N, self.encoder_input_length, self.embedding_size]

        # Last PRENORM
        encoder_input_embedding = tf.contrib.layers.layer_norm(encoder_input_embedding, begin_norm_axis=2)
        encoder_input_embedding = tf.layers.dense(encoder_input_embedding, units=self.embedding_size, activation=None,
                                                  use_bias=False)
        encoder_input_embedding = tf.contrib.layers.layer_norm(encoder_input_embedding, begin_norm_axis=2)
        return encoder_input_embedding  # [N, self.encoder_input_length, self.embedding_size]

    def multi_head_attention_add_norm(self, query, key_value, score_mask=None, output_mask=None, activation=None,
                                      name=None):
        # Sharing Variables
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # for문으로 self.multihead_num번 돌릴 필요 없이 embedding_size 만큼 만들고 self.multihead_num등분해서 연산하면 됨.
            V = tf.layers.dense(  # layers dense는 배치(N)별로 동일하게 연산됨.
                key_value,
                units=self.hidden_size,
                activation=activation,
                use_bias=False,
                name='V',
                # kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
            )  # [N, key_value_sequence_length, self.embedding_size]
            K = tf.layers.dense(
                key_value,
                units=self.hidden_size,
                activation=activation,
                use_bias=False,
                name='K',
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )  # [N, key_value_sequence_length, self.embedding_size]
            Q = tf.layers.dense(
                query,
                units=self.hidden_size,
                activation=activation,
                use_bias=False,
                name='Q',
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
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
            score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.hidden_size / self.multihead_num)

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
                units=self.hidden_size,
                activation=activation,
                use_bias=False,
                name='linear',
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )  # [N, query_sequence_length, self.embedding_size]

            if output_mask is not None:
                Multihead *= output_mask

            # residual Drop Out
            Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
            # Add
            Multihead += query

            return Multihead

    def dense_add_norm(self, embedding, units, output_mask=None, activation=None, name=None):
        # FFN(x) = max(0, x*W1+b1)*W2 + b2
        # Sharing Variables
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            inner_layer = tf.layers.dense(
                embedding,
                units=4 * self.hidden_size,  # bert paper
                activation=activation,  # relu
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )  # [N, self.decoder_input_length, 4*self.embedding_size]
            dense = tf.layers.dense(
                inner_layer,
                units=units,
                activation=None,
                # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )  # [N, self.decoder_input_length, self.embedding_size]

            if output_mask is not None:
                dense *= output_mask  # set 0 bias added pad position

            # Drop out
            dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
            # Add
            dense += embedding

        return dense

    def gelu(self, x):  # https://github.com/google-research/bert/blob/master/modeling.py
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
            x: float Tensor to perform activation.

        Returns:
            `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf