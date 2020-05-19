
import os
import sys
import random
from util import load_json, LabelEncoder

class Session:
    def __init__(self, dataset, label_encoder=None):
        self.dataset = dataset
        self.model_input = []
        self.model_input_A_length = []
        self.positive_label = []
        self.negative_label = []

        if label_encoder:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = LabelEncoder(unk_token = '@unk')

        self.total_songs = []
        self.total_tags = []
        self.others_for_encoder = ['@cls', '@sep', '@mask', '@pad'] # TODO mask는 안쓰면 나중에 지우자.


    def set_label_encoder(self):
        temp_tags = set()
        temp_songs = set()

        for each in self.dataset:
            plylst_tags = each['tags']  # list
            songs = each['songs']  # song id list

            temp_tags.update(plylst_tags)
            temp_songs.update(songs)

        self.label_encoder.fit(list(temp_songs) + list(temp_tags) + self.others_for_encoder)
        self.total_tags = list(temp_tags)
        self.total_songs = list(temp_songs)

    def convert_model_input(self, songs, tags):
        result = ['@cls']
        if songs:
            result += songs
        result += ['@sep']
        if tags:
            result += tags
        result += ['@sep']
        return result


    def make_next_k_song_data(self, songs, tags, k=3, negative_k=10, sample_num=1000):
        max_songs = 100 - k
        max_tags = 5  # val, test set에 최대 tag가 5개 있음.

        result = {}
        if len(songs) < k:
            return {}

        for _ in range(sample_num):

            # sample song이 0개이면 tag는 1개 이상은 있어야하고
            # tag가 0개이면 sample song이 1개 이상은 있어야 함.

            sample_songs = []
            sample_songs_num = random.randint(0, min(len(songs) - k, max_songs))
            if sample_songs_num != 0:
                start_index = random.randint(0, len(songs) - k - sample_songs_num)
                sample_songs = songs[start_index:start_index + sample_songs_num]
                next_songs = songs[start_index + sample_songs_num:start_index + sample_songs_num + k]
            else:
                next_songs = songs[:k]

            negative_songs = []
            while True:
                negative_song = random.choice(self.total_songs)
                if negative_song not in songs and negative_song not in negative_songs:  # plylst에 없는 negative sample인 경우
                    negative_songs.append(negative_song)
                if len(negative_songs) == negative_k:
                    break

            min_tags = 0
            if sample_songs_num == 0:
                min_tags = 1

            if len(tags) < min_tags:
                continue

            sample_tags = []
            sample_tags_num = random.randint(min_tags, min(len(tags), max_tags))
            if sample_tags_num != 0:
                sample_tags = sorted(random.sample(tags, sample_tags_num))  # 중복 데이터셋 처리할 때 편하려고

            model_input = tuple(self.convert_model_input(sample_songs, sample_tags))
            if model_input not in result:
                model_input_A_length = len(sample_songs) + 2 # cls || sample_songs || sep
                result[model_input] = [next_songs, negative_songs, model_input_A_length]

        return result


    def make_dataset(self, positive_k=3, negative_k=10, sample_num_of_each_plylst=1000):
        for each in self.dataset:
            plylst_tags = each['tags']  # list
            songs = each['songs']  # song id list

            next_k_song_data = self.make_next_k_song_data(
                songs,
                plylst_tags,
                k=positive_k,
                negative_k=negative_k,
                sample_num=sample_num_of_each_plylst)

            for data in next_k_song_data:
                positive, negative, model_input_A_length = next_k_song_data[data]
                self.model_input.append(self.label_encoder.transform(data))
                self.positive_label.append(self.label_encoder.transform(positive))
                self.negative_label.append(self.label_encoder.transform(negative))
                self.model_input_A_length.append(model_input_A_length)

            # print(next_k_song_data)
            # print(dataset[-1])
            # print(positive_label[-1])
            # print(negative_label[-1])
            # if input() == " ":
            #     break

if __name__ == "__main__":
    train_set = load_json(os.path.join(workspace_path, 'dataset/orig/train.json'))
    train_session = Session(train_set)
    train_session.set_label_encoder()
    train_session.make_dataset(positive_k=3, negative_k=10, sample_num_of_each_plylst=1)

    val_set = load_json(os.path.join(workspace_path, 'dataset/orig/val.json'))
    val_session = Session(val_set, train_session.label_encoder)
    val_session.set_label_encoder()

    # make_next_k_song_data(songs=[1,2,3,4,5,6,7,8], tags=[11,12,13,14,15], k = 3, negative_k = 10, sample_num=1)

