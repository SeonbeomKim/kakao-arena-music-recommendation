# coding=utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        with tf.name_scope("multi-head-att") as scope:
            # x.shape = [batch_size, seq_len, embedding_dim]
            batch_size = tf.shape(inputs)[0]
            query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
            key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
            value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
            query = self.separate_heads(
                query, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
            key = self.separate_heads(
                key, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
            value = self.separate_heads(
                value, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
            attention, weights = self.attention(query, key, value)
            attention = tf.transpose(
                attention, perm=[0, 2, 1, 3]
            )  # (batch_size, seq_len, num_heads, projection_dim)
            concat_attention = tf.reshape(
                attention, (batch_size, -1, self.embed_dim)
            )  # (batch_size, seq_len, embed_dim)
            output = self.combine_heads(
                concat_attention
            )  # (batch_size, seq_len, embed_dim)
            return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        with tf.name_scope('transformer_block') as scope:
            attn_output = self.att(inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, emded_dim):
        super(TokenEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)

    def call(self, x):
        with tf.name_scope('embedding') as scope:
            x = self.token_emb(x)
            return x


class Classifier(object):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.dropout_rate = rate
        self.target_vocab_size = target_vocab_size

        self.embedding_layer = TokenEmbedding(self.input_vocab_size, self.d_model)
        self.transformer_block = TransformerBlock(self.d_model, self.num_heads, self.dff)

    def build_model(self, max_seq, fc_size, training):
        with tf.name_scope("classifier_keras") as scope:
            inputs = keras.layers.Input(shape=(max_seq,))
            emb = self.embedding_layer(inputs)
            trans = self.transformer_block(emb, training)
            avg_pool = layers.GlobalAveragePooling1D()(trans)
            do1 = layers.Dropout(self.dropout_rate)(avg_pool)
            fc = layers.Dense(fc_size, activation="relu", name="output_fc")(do1)
            do2 = layers.Dropout(self.dropout_rate)(fc)
            outputs = layers.Dense(self.target_vocab_size, activation="sigmoid", name="output")(do2)

            model = keras.Model(inputs=inputs, outputs=outputs)
            return model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TagClassifierDataGenerator(tf.keras.utils.Sequence):
    """
    referenced from
    https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

    """
    def __init__(self, song_list, tag_list,
                 song_encoder: tf.keras.preprocessing.text.Tokenizer,
                 tag_encoder: sklearn.preprocessing.MultiLabelBinarizer,
                 max_song_seq, max_tag_size, batch_size,
                 training=True, shuffle=True):
        self.song_list = song_list
        self.tag_list = tag_list
        self.song_encoder = song_encoder
        self.tag_encoder = tag_encoder
        self.max_song_seq = max_song_seq
        self.max_tag_size = max_tag_size
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle
        self.indices = np.arange(len(self.song_list))

        self.on_epoch_end()

        assert(len(self.song_list) == len(self.tag_list))

    def __len__(self):
        return int(np.floor(len(self.song_list) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indices = np.arange(len(self.song_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        batch_x = self._get_batch_x(batch_indices)

        if self.training:
            batch_y = self._get_batch_y(batch_indices)
            return batch_x, batch_y
        else:
            return batch_x

    def _get_batch_x(self, batch_indices):
        """Generates data containing batch_size images
        :param batch_indices: list of label ids to load
        :return: batch of x input
        """
        # Initialization
        batch_x = np.empty((self.batch_size, self.max_song_seq))

        # Generate data
        for i, data_id in enumerate(batch_indices):
            # Store sample
            song_seq = self.song_encoder.texts_to_sequences([self.song_list[data_id]])
            batch_x[i, ] = tf.keras.preprocessing.sequence.pad_sequences(song_seq, maxlen=self.max_song_seq)[0]

        return batch_x

    def _get_batch_y(self, batch_indices):
        """Generates data containing batch_size masks
        :param batch_indices: list of label ids to load
        :return: batch of y input
        """
        batch_y = np.empty((self.batch_size, self.max_tag_size), dtype=int)

        # Generate data
        for i, data_id in enumerate(batch_indices):
            # Store sample
            batch_y[i, ] = self.tag_encoder.transform([self.tag_list[data_id]])[0]

        return batch_y


def loss_function(real, pred, pos_loss_weight=0.9, neg_loss_weight=0.1):
    # print(f"real: {real.shape}")
    # print(f"pred: {pred.shape}")
    positive_mask = tf.math.logical_not(tf.math.equal(real, 0))
    positive_mask = tf.cast(positive_mask, dtype=pred.dtype)
#     print(f"positive_mask: {positive_mask}")
    positive_loss = tf.math.log(pred + tf.keras.backend.epsilon())

    negative_mask = tf.math.logical_not(tf.math.equal(real, 1))
    negative_mask = tf.cast(negative_mask, dtype=pred.dtype)
#     print(f"negative_mask: {negative_mask}")
    negative_loss = tf.math.log(1 - pred + tf.keras.backend.epsilon())
#     loss_ = loss_object(real, pred)
    loss_ = pos_loss_weight * positive_mask * positive_loss + neg_loss_weight * negative_mask * negative_loss
    loss_ *= -1
    # ret = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return tf.reduce_sum(loss_) / len(real)
