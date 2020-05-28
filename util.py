import json
import os
import pickle
import random

import numpy as np
import pandas as pd


def select_bucket(data_size, bucket_size):
    for bucket in bucket_size:
        if data_size <= bucket:
            return bucket
    return None


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


def fill_na(data, fill_value=0):
    # data:[[1,2],[1],[1,2,3]]
    # return: [[1,2,fill_value], [1,fill_value,fill_value], [1,2,3]]
    df = pd.DataFrame(data)
    return df.fillna(fill_value).values  # numpy type


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
