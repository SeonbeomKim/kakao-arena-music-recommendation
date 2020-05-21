import os
import random

import numpy as np
import util
from tqdm import tqdm


class LabelEncoder:
    def __init__(self, dataset):
        self.total_songs = []
        self.total_tags = []
        self.others_for_encoder = ['@cls', '@sep', '@mask', '@pad']  # TODO mask는 안쓰면 나중에 지우자.
        self.label_encoder = util.LabelEncoder(unk_token='@unk')
        self.set_label_encoder(dataset)

    def set_label_encoder(self, dataset):
        temp_tags = set()
        temp_songs = set()

        for each in dataset:
            plylst_tags = each['tags']  # list
            songs = each['songs']  # song id list

            temp_tags.update(plylst_tags)
            temp_songs.update(songs)

        self.total_tags = list(temp_tags)
        self.total_songs = list(temp_songs)
        self.label_encoder.fit(self.total_songs + self.total_tags + self.others_for_encoder)


def convert_model_input(songs, tags, label_encoder=None):
    result = ['@cls']
    if songs:
        result += songs
    result += ['@sep']
    if tags:
        result += tags
    result += ['@sep']

    if label_encoder:
        result = label_encoder.transform(result)
    return result


def make_next_k_song_data(songs, tags, total_songs, k=3, negative_k=10, sample_num=1000):
    max_songs = 99  # 노래는 100개 예측해야하는데, 99개 들어왔을 때 나머지 1개 예측하도록 처리하는게 가장 큰 인풋.
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
            negative_song = random.choice(total_songs)
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

        model_input = tuple(convert_model_input(sample_songs, sample_tags))
        if model_input not in result:
            model_input_A_length = len(sample_songs) + 2  # cls || sample_songs || sep
            result[model_input] = [next_songs, negative_songs, model_input_A_length]

    return result


def make_model_dataset(dataset, label_encoder, total_songs, positive_k=3, negative_k=10,
                       sample_num_of_each_plylst=1000):
    model_input = []
    positive_label = []
    negative_label = []
    model_input_A_length = []

    for each in tqdm(dataset, total=len(dataset)):
        plylst_tags = each['tags']  # list
        songs = each['songs']  # song id list

        next_k_song_data = make_next_k_song_data(
            songs,
            plylst_tags,
            total_songs,
            k=positive_k,
            negative_k=negative_k,
            sample_num=sample_num_of_each_plylst)

        for data in next_k_song_data:
            positive, negative, A_length = next_k_song_data[data]
            model_input.append(label_encoder.transform(data))
            positive_label.append(label_encoder.transform(positive))
            negative_label.append(label_encoder.transform(negative))
            model_input_A_length.append(A_length)

    # dataset to numpy
    model_input = np.array(model_input)  # row마다 column수가 달라서 dtype 줄 수 없음
    model_input_A_length = np.array(model_input_A_length, np.int32)
    positive_label = np.array(positive_label, np.int32)
    negative_label = np.array(negative_label, np.int32)

    model_dataset = {
        'model_input': model_input,
        'model_input_A_length': model_input_A_length,
        'positive_label': positive_label,
        'negative_label': negative_label}

    return model_dataset


def select_bucket(data_size, bucket_size):
    for bucket in bucket_size:
        if data_size <= bucket:
            return bucket
    return None


def make_model_dataset_bucket(dataset, label_encoder, total_songs, positive_k=3, negative_k=10,
                              sample_num_of_each_plylst=1000, bucket_size=[50, 107], pad_symbol='@pad'):
    bucket_dataset = {
        bucket: {'model_input': [], 'positive_label': [], 'negative_label': [], 'model_input_A_length': []} for bucket
        in bucket_size}

    for each in tqdm(dataset, total=len(dataset)):
        plylst_tags = each['tags']  # list
        songs = each['songs']  # song id list

        next_k_song_data = make_next_k_song_data(
            songs,
            plylst_tags,
            total_songs,
            k=positive_k,
            negative_k=negative_k,
            sample_num=sample_num_of_each_plylst)

        for data in next_k_song_data:
            positive, negative, A_length = next_k_song_data[data]
            data_size = len(data)
            bucket = select_bucket(data_size, bucket_size)
            if not bucket:
                continue

            data = list(data) + [pad_symbol] * (bucket - data_size)
            bucket_dataset[bucket]['model_input'].append(label_encoder.transform(data))
            bucket_dataset[bucket]['positive_label'].append(label_encoder.transform(positive))
            bucket_dataset[bucket]['negative_label'].append(label_encoder.transform(negative))
            bucket_dataset[bucket]['model_input_A_length'].append(A_length)

    # dataset to numpy
    for bucket in bucket_dataset:
        bucket_dataset[bucket]['model_input'] = np.array(bucket_dataset[bucket]['model_input'], np.int32)
        bucket_dataset[bucket]['positive_label'] = np.array(bucket_dataset[bucket]['positive_label'], np.int32)
        bucket_dataset[bucket]['negative_label'] = np.array(bucket_dataset[bucket]['negative_label'], np.int32)
        bucket_dataset[bucket]['model_input_A_length'] = np.array(bucket_dataset[bucket]['model_input_A_length'],
                                                                  np.int32)
    return bucket_dataset

#
# class Session:
#     def __init__(self, dataset, label_encoder=None):
#         self.dataset = dataset
#         self.model_input = []
#         self.model_input_A_length = []
#         self.positive_label = []
#         self.negative_label = []
#
#         if label_encoder:
#             self.label_encoder = label_encoder
#         else:
#             self.label_encoder = util.LabelEncoder(unk_token='@unk')
#
#         self.total_songs = []
#         self.total_tags = []
#         self.others_for_encoder = ['@cls', '@sep', '@mask', '@pad']  # TODO mask는 안쓰면 나중에 지우자.
#
#     def set_label_encoder(self):
#         temp_tags = set()
#         temp_songs = set()
#
#         for each in self.dataset:
#             plylst_tags = each['tags']  # list
#             songs = each['songs']  # song id list
#
#             temp_tags.update(plylst_tags)
#             temp_songs.update(songs)
#
#         self.label_encoder.fit(list(temp_songs) + list(temp_tags) + self.others_for_encoder)
#         self.total_tags = list(temp_tags)
#         self.total_songs = list(temp_songs)
#
#     def convert_model_input(self, songs, tags):
#         result = ['@cls']
#         if songs:
#             result += songs
#         result += ['@sep']
#         if tags:
#             result += tags
#         result += ['@sep']
#         return result
#
#     def make_next_k_song_data(self, songs, tags, k=3, negative_k=10, sample_num=1000):
#         max_songs = 99 # 노래는 100개 예측해야하는데, 99개 들어왔을 때 나머지 1개 예측하도록 처리하는게 가장 큰 인풋.
#         max_tags = 5  # val, test set에 최대 tag가 5개 있음.
#
#         result = {}
#         if len(songs) < k:
#             return {}
#
#         for _ in range(sample_num):
#
#             # sample song이 0개이면 tag는 1개 이상은 있어야하고
#             # tag가 0개이면 sample song이 1개 이상은 있어야 함.
#
#             sample_songs = []
#             sample_songs_num = random.randint(0, min(len(songs) - k, max_songs))
#             if sample_songs_num != 0:
#                 start_index = random.randint(0, len(songs) - k - sample_songs_num)
#                 sample_songs = songs[start_index:start_index + sample_songs_num]
#                 next_songs = songs[start_index + sample_songs_num:start_index + sample_songs_num + k]
#             else:
#                 next_songs = songs[:k]
#
#             negative_songs = []
#             while True:
#                 negative_song = random.choice(self.total_songs)
#                 if negative_song not in songs and negative_song not in negative_songs:  # plylst에 없는 negative sample인 경우
#                     negative_songs.append(negative_song)
#                 if len(negative_songs) == negative_k:
#                     break
#
#             min_tags = 0
#             if sample_songs_num == 0:
#                 min_tags = 1
#
#             if len(tags) < min_tags:
#                 continue
#
#             sample_tags = []
#             sample_tags_num = random.randint(min_tags, min(len(tags), max_tags))
#             if sample_tags_num != 0:
#                 sample_tags = sorted(random.sample(tags, sample_tags_num))  # 중복 데이터셋 처리할 때 편하려고
#
#             model_input = tuple(self.convert_model_input(sample_songs, sample_tags))
#             if model_input not in result:
#                 model_input_A_length = len(sample_songs) + 2  # cls || sample_songs || sep
#                 result[model_input] = [next_songs, negative_songs, model_input_A_length]
#
#         return result
#
#     def make_dataset(self, positive_k=3, negative_k=10, sample_num_of_each_plylst=1000):
#         for each in self.dataset:
#             plylst_tags = each['tags']  # list
#             songs = each['songs']  # song id list
#
#             next_k_song_data = self.make_next_k_song_data(
#                 songs,
#                 plylst_tags,
#                 k=positive_k,
#                 negative_k=negative_k,
#                 sample_num=sample_num_of_each_plylst)
#
#             for data in next_k_song_data:
#                 positive, negative, model_input_A_length = next_k_song_data[data]
#                 self.model_input.append(self.label_encoder.transform(data))
#                 self.positive_label.append(self.label_encoder.transform(positive))
#                 self.negative_label.append(self.label_encoder.transform(negative))
#                 self.model_input_A_length.append(model_input_A_length)
#
#             # print(next_k_song_data)
#             # print(dataset[-1])
#             # print(positive_label[-1])
#             # print(negative_label[-1])
#             # if input() == " ":
#             #     break
#
#
# if __name__ == "__main__":
#     train_set = load_json(os.path.join(workspace_path, 'dataset/orig/train.json'))
#     train_session = Session(train_set)
#     train_session.set_label_encoder()
#     train_session.make_dataset(positive_k=3, negative_k=10, sample_num_of_each_plylst=1)
#
#     val_set = load_json(os.path.join(workspace_path, 'dataset/orig/val.json'))
#     val_session = Session(val_set, train_session.label_encoder)
#     val_session.set_label_encoder()
#
#     # make_next_k_song_data(songs=[1,2,3,4,5,6,7,8], tags=[11,12,13,14,15], k = 3, negative_k = 10, sample_num=1)
