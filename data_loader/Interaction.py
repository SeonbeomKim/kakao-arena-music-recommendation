import json

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class Interaction(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.raw_tag_playlist_ids = []
        self.encoded_tag_playlist_ids = []
        self.tag_playlist_ids_set = set()

        self.raw_song_playlist_ids = []
        self.encoded_song_playlist_ids = []
        self.song_playlist_ids_set = set()

        self.non_interaction_playlist_set = set()

        self.raw_song = []
        self.encoded_song = []
        self.song_set = set()

        self.raw_tags = []
        self.encoded_tags = []
        self.tag_set = set()

        self.playlist_set = None

        self.playlist_encoder = LabelEncoder()
        self.tag_encoder = LabelEncoder()
        self.song_encoder = LabelEncoder()

        self.process_train_data()
        self.encode_train_ids()

    def process_train_data(self):
        for playlist_data in self.train_data:
            playlist_id = playlist_data['id']
            tags = playlist_data['tags']
            songs = playlist_data['songs']
            if len(tags) == 0:
                self.non_interaction_playlist_set.add(playlist_id)
            if len(songs) == 0:
                self.non_interaction_playlist_set.add(playlist_id)
            self.raw_tag_playlist_ids.extend([playlist_id] * len(tags))
            self.raw_song_playlist_ids.extend([playlist_id] * len(songs))
            self.raw_tags.extend(tags)
            self.raw_song.extend(songs)

    def encode_train_ids(self):
        # encode only train data
        self.tag_playlist_ids_set = set(self.raw_tag_playlist_ids)
        self.song_playlist_ids_set = set(self.raw_song_playlist_ids)
        self.playlist_set = self.tag_playlist_ids_set | self.song_playlist_ids_set | self.non_interaction_playlist_set
        self.playlist_encoder.fit_transform(list(self.playlist_set))
        self.tag_set = set(self.raw_tags)
        self.song_set = set(self.raw_song)
        self.encoded_tag_playlist_ids = self.playlist_encoder.transform(self.raw_tag_playlist_ids)
        self.encoded_song_playlist_ids = self.playlist_encoder.transform(self.raw_song_playlist_ids)
        self.encoded_tags = self.tag_encoder.fit_transform(self.raw_tags)
        self.encoded_song = self.song_encoder.fit_transform(self.raw_song)

    def build_playlist_tag_matrix(self):
        # TODO: check this in an elegant way
        assert len(self.encoded_tags) == len(self.encoded_tag_playlist_ids)
        data = np.ones((len(self.encoded_tags)))
        return csr_matrix((data, (self.encoded_tag_playlist_ids, self.encoded_tags)))

    def build_playlist_song_matrix(self):
        assert len(self.encoded_song) == len(self.encoded_song_playlist_ids)
        data = np.ones((len(self.encoded_song)))
        return csr_matrix((data, (self.encoded_song_playlist_ids, self.encoded_song)))

    @property
    def num_playlist_ids(self):
        return len(set(self.raw_tag_playlist_ids))

    @property
    def num_tags(self):
        return len(set(self.encoded_tags))


if __name__ == "__main__":
    song_meta_file = "dataset/orig/song_meta.json"
    train_file = "dataset/orig/train.json"
    val_file = "dataset/orig/val.json"
    with open(train_file) as f:
        train_data = json.load(f)
    with open(train_file) as f:
        val_data = json.load(f)

    interaction = Interaction(train_data)
    train_matrix = interaction.build_playlist_tag_matrix()

    print("finish")
