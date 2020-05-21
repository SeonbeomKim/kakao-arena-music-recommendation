import os
import random

import numpy as np
import util
from tqdm import tqdm


class LabelEncoder:
    def __init__(self, dataset, songs_recall=0.9, tags_recall=0.95):
        self.songs = []
        self.tags = []
        self.others_for_encoder = ['@cls', '@sep', '@mask', '@pad']  # TODO mask는 안쓰면 나중에 지우자.
        self.unk_token = '@unk'
        self.label_encoder = util.LabelEncoder(unk_token=self.unk_token)
        self.set_label_encoder(dataset, songs_recall=0.9, tags_recall=0.95)

    def filter_row_freq_item(self, dataset, item_key, recall=0.9):
        freq_dict = {}

        total_freq = 0
        for each in dataset:
            for item in each[item_key]:
                if item not in freq_dict:
                    freq_dict[item] = 0
                freq_dict[item] += 1
                total_freq += 1

        print('%s num: %d:' % (item_key, len(freq_dict)))
        freq_sorted_items = sorted(list(freq_dict.items()), key=lambda each:each[1], reverse=True)

        accum_freq = 0
        slice_idx = 0
        for idx, item_freq in enumerate(freq_sorted_items):
            if (accum_freq / float(total_freq)) > recall:
                slice_idx = idx
                break
            song, freq = item_freq
            accum_freq += freq

        filtered_items = [item_freq[0] for item_freq in freq_sorted_items[:slice_idx]]
        print('%s num(after filtering): %d, freq: %d' % (item_key, len(filtered_items), freq_sorted_items[slice_idx-1][1]))
        return filtered_items

    def set_label_encoder(self, dataset, songs_recall=0.9, tags_recall=0.95):
        self.songs = self.filter_row_freq_item(dataset, 'songs', recall=songs_recall)
        self.tags = self.filter_row_freq_item(dataset, 'tags', recall=tags_recall)
        self.label_encoder.fit(self.songs + self.tags + self.others_for_encoder)


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


        # TODO Hard negative 뽑도록 변경해야함
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


def select_bucket(data_size, bucket_size):
    for bucket in bucket_size:
        if data_size <= bucket:
            return bucket
    return None

def is_all_unk(data, unk_idx, is_model_input=False):
    # data: idx로 변환된 tokens if is_model_input: cls || songs || sep || tags || sep
    data_size = len(data)
    if is_model_input:
        data_size -= 3 # cls,sep,sep 제외
    if data.count(unk_idx) == data_size:
        return True
    return False

def make_model_dataset_bucket(dataset, label_encoder, total_songs, positive_k=3, negative_k=10,
                              sample_num_of_each_plylst=1000, bucket_size=[50, 107], pad_symbol='@pad', unk_symbol='@unk'):
    unk_idx = label_encoder.transform([unk_symbol])[0]
    pad_idx = label_encoder.transform([pad_symbol])[0]

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

            data = label_encoder.transform(list(data))
            if is_all_unk(data, unk_idx, is_model_input=True):
                continue
            data += [pad_idx] * (bucket - data_size)

            positive = label_encoder.transform(positive)
            if is_all_unk(positive, unk_idx, is_model_input=False):
                continue

            negative = label_encoder.transform(negative)

            bucket_dataset[bucket]['model_input'].append(data)
            bucket_dataset[bucket]['positive_label'].append(positive)
            bucket_dataset[bucket]['negative_label'].append(negative)
            bucket_dataset[bucket]['model_input_A_length'].append(A_length)

    # dataset to numpy
    for bucket in bucket_dataset:
        bucket_dataset[bucket]['model_input'] = np.array(bucket_dataset[bucket]['model_input'], np.int32)
        bucket_dataset[bucket]['positive_label'] = np.array(bucket_dataset[bucket]['positive_label'], np.int32)
        bucket_dataset[bucket]['negative_label'] = np.array(bucket_dataset[bucket]['negative_label'], np.int32)
        bucket_dataset[bucket]['model_input_A_length'] = np.array(bucket_dataset[bucket]['model_input_A_length'],
                                                                  np.int32)
    return bucket_dataset

