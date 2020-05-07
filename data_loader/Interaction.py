import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix


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


if __name__=="__main__":
    song_meta_file = "../res/song_meta.json"
    train_file = "../res/train.json"
    with open(train_file) as f:
        data = json.load(f)
    interaction = Interaction(data)
    matrix = interaction.build_interaction_matrix()
    print("finish")
