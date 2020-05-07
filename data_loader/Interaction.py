import numpy as np
import json


class Interaction(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.playlist_ids = []
        self.song_ids = []


if __name__=="__main__":
    song_meta_file = "../res/song_meta.json"
    train_file = "../res/train.json"
    with open(train_file) as f:
        data = json.load(f)
    print(data)
