import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse as sp


@tf.function
def get_label(co_occur):
    label = tf.cast(co_occur > 0, tf.float32)
    return label


@tf.function
def get_confidence(co_occur, alpha=40.):
    confidence = 1 + alpha * co_occur
    return confidence


def process_data(dataset, neg_dataset):
    neg_dataset_data = np.array(neg_dataset.size * [0])
    total_rows = np.concatenate([dataset.row, neg_dataset.row])
    total_cols = np.concatenate([dataset.col, neg_dataset.col])
    total_data = np.concatenate([dataset.data, neg_dataset_data])

    idx = np.random.permutation(total_rows)
    playlist_ids = total_rows[idx].astype(np.int32)
    song_ids = total_cols[idx].astype(np.int32)
    co_occurs = total_data[idx].astype(np.float32)
    return playlist_ids, song_ids, co_occurs


class WMF:
    def __init__(self, U_size, V_size, embedding_size, regularization_factor, alpha=40., learning_rate=0.05):
        self.U_size = U_size
        self.V_size = V_size
        self.embedding_size = embedding_size
        self.num_training = None
        # TODO: loss object 맞는지 확인 필요
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = None
        self.learning_rate = learning_rate
        self.train_loss_estimator = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_estimator = tf.keras.metrics.Mean(name="val_loss")
        self.test_loss_estimator = tf.keras.metrics.Mean(name="test_loss")
        self.regularization_factor = regularization_factor
        self.alpha = alpha

        self.U_embedding_table = None
        self.V_embedding_table = None
        self.U_bias = None
        self.V_bias = None

    def build_network(self):
        self.U_embedding_table = tf.Variable(
            tf.random.truncated_normal((self.U_size, self.embedding_size), stddev=0.02),
            name="U_embedding_table")

        self.V_embedding_table = tf.Variable(
            tf.random.truncated_normal((self.V_size, self.embedding_size), stddev=0.02),
            name="V_embedding_table")

        self.U_bias = tf.Variable(
            tf.random.truncated_normal((self.U_size, 1), stddev=0.02),
            name="U_bias")

        self.V_bias = tf.Variable(
            tf.random.truncated_normal((self.V_size, 1), stddev=0.02),
            name="V_bias")

        self.trainable_variables = [
            self.U_embedding_table, self.V_embedding_table, self.U_bias, self.V_bias
        ]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    # @tf.function
    def predict(self, U_idx, V_idx):
        U_init_embedding = tf.nn.embedding_lookup(
            self.U_embedding_table,
            U_idx)
        U_bias_embedding = tf.nn.embedding_lookup(
            self.U_bias,
            U_idx)
        V_init_embedding = tf.nn.embedding_lookup(
            self.V_embedding_table,
            V_idx)
        V_bias_embedding = tf.nn.embedding_lookup(
            self.V_bias,
            V_idx)
        U_embedding = U_init_embedding + U_bias_embedding
        V_embedding = V_init_embedding + V_bias_embedding

        predict = tf.reduce_sum(U_embedding * V_embedding, axis=-1)
        return predict

    def cal_reg(self, U_idx, V_idx):
        U_init_embedding = tf.nn.embedding_lookup(
            self.U_embedding_table,
            U_idx)
        U_bias_embedding = tf.nn.embedding_lookup(
            self.U_bias,
            U_idx)
        V_init_embedding = tf.nn.embedding_lookup(
            self.V_embedding_table,
            V_idx)
        V_bias_embedding = tf.nn.embedding_lookup(
            self.V_bias,
            V_idx)

        reg_embedding = tf.reduce_sum(tf.nn.l2_loss(U_init_embedding)) + tf.reduce_sum(tf.nn.l2_loss(V_init_embedding))
        reg_bias = tf.reduce_sum(tf.nn.l2_loss(U_bias_embedding)) + tf.reduce_sum(tf.nn.l2_loss(V_bias_embedding))
        reg_loss = reg_embedding + reg_bias
        return reg_loss

    # @tf.function
    def train_op(self, batch_user_ids, batch_item_ids, batch_co_occurs):
        with tf.GradientTape() as tape:
            pred_y = self.predict(batch_user_ids, batch_item_ids)
            labels = get_label(batch_co_occurs)
            confidence = get_confidence(batch_co_occurs)
            reg_loss = self.cal_reg(batch_user_ids, batch_item_ids)
            loss = self.loss_object(labels, pred_y) * confidence + reg_loss
        gradient_of_model = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_of_model, self.trainable_variables))
        self.train_loss_estimator(loss)

    def measure_op(self, batch_user_ids, batch_item_ids, batch_co_occurs, mode="test"):
        pred_y = self.predict(batch_user_ids, batch_item_ids)
        labels = get_label(batch_co_occurs)
        confidence = get_confidence(batch_co_occurs)
        reg_loss = self.cal_reg(batch_user_ids, batch_item_ids)
        loss = self.loss_object(labels, pred_y) * confidence + reg_loss
        if mode == "test":
            self.test_loss_estimator(loss)
        elif mode == "val":
            self.val_loss_estimator(loss)
        else:
            raise ValueError("Invalid mode Value")

    def train(self, dataset, neg_dataset, batch_size=1024, epochs=150):
        train_dataset, valid_dataset = train_test_split(
            dataset, test_size=0.2, random_state=42)
        train_dataset, valid_dataset = train_dataset.tocoo(), valid_dataset.tocoo()

        train_neg_dataset, valid_neg_dataset = train_test_split(
            neg_dataset, test_size=0.2, random_state=42)
        train_neg_dataset, valid_neg_dataset = train_neg_dataset.tocoo(), valid_neg_dataset.tocoo()

        val_playlist_ids, val_song_ids, val_co_occurs = process_data(valid_dataset, valid_neg_dataset)
        for epoch in range(epochs):
            # Concatenate and shuffling data
            train_playlist_ids, train_song_ids, train_co_occurs = process_data(train_dataset, train_neg_dataset)

            num_train_data = len(train_playlist_ids)
            train_total_batch = (num_train_data // batch_size) + 1
            num_val_data = len(val_playlist_ids)
            val_total_batch = (num_val_data // batch_size) + 1

            # train
            for i in range(train_total_batch):
                batch_playlist_ids = train_playlist_ids[i * batch_size: min((i + 1) * batch_size, num_train_data)]
                batch_song_ids = train_song_ids[i * batch_size: min((i + 1) * batch_size, num_train_data)]
                batch_co_occurs = train_co_occurs[i * batch_size: min((i + 1) * batch_size, num_train_data)]
                self.train_op(batch_playlist_ids, batch_song_ids, batch_co_occurs)
                # TODO: Metric 설정 및 추가 해야 함.

            # validation
            for i in range(val_total_batch):
                batch_playlist_ids = val_playlist_ids[i * batch_size: min((i + 1) * batch_size, num_val_data)]
                batch_song_ids = val_song_ids[i * batch_size: min((i + 1) * batch_size, num_val_data)]
                batch_co_occurs = val_co_occurs[i * batch_size: min((i + 1) * batch_size, num_val_data)]
                self.measure_op(batch_playlist_ids, batch_song_ids, batch_co_occurs, mode="val")
            print(f"train epoch : {epoch} loss: {self.train_loss_estimator.result()} val loss : {self.val_loss_estimator.result()}")
            self.train_loss_estimator.reset_states()
            self.val_loss_estimator.reset_states()

    def test(self, dataset, neg_dataset, batch_size=512):
        num_data = dataset.shape[0]
        total_batch = (num_data // batch_size) + 1
        playlist_ids, song_ids, co_occurs = process_data(dataset, neg_dataset)

        for i in range(total_batch):
            batch_playlist_ids = playlist_ids[i * batch_size: min((i + 1) * batch_size, num_data)]
            batch_song_ids = song_ids[i * batch_size: min((i + 1) * batch_size, num_data)]
            batch_co_occurs = co_occurs[i * batch_size: min((i + 1) * batch_size, num_data)]
            self.measure_op(batch_playlist_ids, batch_song_ids, batch_co_occurs)
        print(f" test loss: {self.test_loss_estimator.result()}")
        self.test_loss_estimator.reset_states()


if __name__ == "__main__":
    wmf = WMF(U_size=200, V_size=300, embedding_size=150, regularization_factor=0.1, alpha=40., learning_rate=0.0001)
    wmf.build_network()