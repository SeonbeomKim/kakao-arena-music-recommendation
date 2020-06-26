import json
import os
import pickle
import random
from datetime import datetime
import parameters
import numpy as np
# import pandas as pd
import re


def get_song_issue_dict(train_set, song_meta, label_info):
    all_songs_set = set(label_info.songs)

    song_issue_dict = {}
    for each in song_meta:
        song_issue_dict[each["id"]] = int(each['issue_date'])

    strange_song_issue_dict = {}
    for each in train_set:
        plylst_updt_date = convert_updt_date(each["updt_date"])
        songs = list(filter(lambda song: song in all_songs_set, each['songs']))
        for song in songs:
            if song_issue_dict[song] <= plylst_updt_date and song_issue_dict[song] != 0:
                continue

            if song not in strange_song_issue_dict:
                strange_song_issue_dict[song] = []
            strange_song_issue_dict[song].append(plylst_updt_date)

    for song in strange_song_issue_dict:
        song_issue_dict[song] = min(strange_song_issue_dict[song])
    return song_issue_dict

def get_song_issue_dict2(train_set, song_meta, label_info):
    all_songs_set = set(label_info.songs)

    song_issue_dict = {}
    for each in song_meta:
        song_issue_dict[each["id"]] = int(each['issue_date'])

    strange_song_issue_dict = {}
    for each in train_set:
        plylst_updt_date = convert_updt_date(each["updt_date"])
        songs = list(filter(lambda song: song in all_songs_set, each['songs']))
        for song in songs:
            if song_issue_dict[song] <= plylst_updt_date:
                continue

            if song not in strange_song_issue_dict:
                strange_song_issue_dict[song] = []
            strange_song_issue_dict[song].append(plylst_updt_date)

    for song in strange_song_issue_dict:
        song_issue_dict[song] = min(strange_song_issue_dict[song])
    return song_issue_dict

def remove_special_char(string):
    return re.sub("[^가-힣a-zA-Z0-9& ]+", ' ', string).strip().lower() # R&B

def convert_updt_date(updt_date):
    dtime = datetime.strptime(updt_date, '%Y-%m-%d %H:%M:%S.%f')
    return int(dtime.strftime("%Y%m%d"))

def get_year_label(updt_date):
    # base_year 미만은 0, base_year는 1, 나머지는 1씩 증가
    updt_date = convert_updt_date(updt_date)
    year = int(str(updt_date)[:4])
    if year < parameters.base_year:
        return '@old'
    else:
        return '@%d' % year

def select_bucket(data_size, bucket_size):
    for bucket in bucket_size:
        if data_size <= bucket:
            return bucket
    return None

def label_to_sparse_label(label):
    sparse_label = []
    for row in range(len(label)):
        for _label in label[row]:
            sparse_label.append([row, _label])
    return sparse_label

def get_k_negative_label(positive_label_set, total_label, k):
    negative_label = []

    for _negative in random.sample(total_label, k + len(positive_label_set)):
        if len(negative_label) == k:
            break
        if _negative in positive_label_set:
            continue
        if _negative in negative_label:
            continue
        negative_label.append(_negative)

    if len(negative_label) != k:
        print('%d != %d' % (len(negative_label), k))
    return negative_label


def dump(data, fname):
    parent = os.path.dirname(fname)
    if not os.path.exists(parent):
        print("create %s" % parent)
        os.makedirs(parent)

    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    if not os.path.exists(parent):
        print("create %s" % parent)
        os.makedirs(parent)

    with open(fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


# def fill_na(data, fill_value=0):
#     # data:[[1,2],[1],[1,2,3]]
#     # return: [[1,2,fill_value], [1,fill_value,fill_value], [1,2,3]]
#     df = pd.DataFrame(data)
#     return df.fillna(fill_value).values  # numpy type


class LabelEncoder:
    def __init__(self, tokens=[], unk_token=''):
        # tokens = ['@cls', '@sep', '@mask', '@pad']
        # unk_token: @unk, deep learning으로 학습할 때, unk 필요한 경우

        self.classes_ = []
        self.data_to_label = {}
        self.tokens = tokens
        self.unk_token = unk_token

    def fit(self, data_list):
        _data_list = data_list[:]  # copy
        if self.tokens:
            _data_list += self.tokens
        if self.unk_token:
            _data_list += [self.unk_token]

        for data in _data_list:
            if data not in self.data_to_label:
                self.data_to_label[data] = len(self.data_to_label)
                self.classes_.append(data)

    def transform(self, data_list):
        trans_data = []

        if self.unk_token:
            unk_label = self.data_to_label[self.unk_token]
        else:
            unk_label = None

        for data in data_list:
            trans_data.append(self.data_to_label.get(data, unk_label))
        return trans_data

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)

    def inverse_transform(self, data_list):
        inverse = []
        for data in data_list:
            inverse.append(self.classes_[data])
        return inverse
