import json

import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def sample_neg(positive_interaction, neg_sample_ratio=1.2):
    num_neg_sample = int(positive_interaction.size * neg_sample_ratio)
    Y = sp.coo_matrix(positive_interaction.shape)
    Y_size = 0
    num_rows, num_cols = positive_interaction.shape
    while Y_size < num_neg_sample:
        num_to_sample = num_neg_sample - Y_size
        data = np.concatenate([Y.data, np.ones(num_to_sample)])
        Y_row = np.concatenate([Y.row, np.random.choice(num_rows, num_to_sample)])
        Y_col = np.concatenate([Y.col, np.random.choice(num_cols, num_to_sample)])
        Y = sp.coo_matrix((data, (Y_row, Y_col)), shape=positive_interaction.shape)
        Y = sp.coo_matrix(Y - positive_interaction.multiply(Y))
        Y_size = Y.size
    return Y


class Interaction(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.playlist_ids = []
        self.song_ids = []
        self.song_encoder = LabelEncoder()
        self.playlist_encoder = LabelEncoder()

    def process_data(self):
        for playlist_data in self.train_data:
            playlist_id = playlist_data['id']
            songs = playlist_data['songs']
            self.playlist_ids.extend([playlist_id] * len(songs))
            self.song_ids.extend(songs)

    def encode_ids(self):
        self.playlist_ids = self.playlist_encoder.fit_transform(self.playlist_ids)
        self.song_ids = self.song_encoder.fit_transform(self.song_ids)

    def build_interaction_matrix(self):
        self.process_data()
        self.encode_ids()
        # TODO: check this in an elegant way
        assert len(self.song_ids) == len(self.playlist_ids)
        data = np.ones((len(self.song_ids)))
        return csr_matrix((data, (self.playlist_ids, self.song_ids)))


if __name__ == "__main__":
    song_meta_file = "../res/song_meta.json"
    train_file = "../res/train.json"
    with open(train_file) as f:
        data = json.load(f)
    interaction = Interaction(data)
    matrix = interaction.build_interaction_matrix()
    neg_matrix = sample_neg(matrix, neg_sample_ratio=1.2)
    print("finish")
