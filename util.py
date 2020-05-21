import json
import os
import pickle

import numpy as np
import pandas as pd


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
    def __init__(self, unk_token='@unk'):
        self.classes_ = None
        self.data_to_label = {}
        self.label_to_data = []
        self.unk_token = unk_token

    def fit(self, data_list):
        for data in data_list + [self.unk_token]:
            if data not in self.data_to_label:
                self.data_to_label[data] = len(self.data_to_label)
                self.label_to_data.append(data)
        self.classes_ = self.label_to_data

    def fit_transform(self, data_list):
        trans_data = []
        for data in data_list + [self.unk_token]:
            if data not in self.data_to_label:
                self.data_to_label[data] = len(self.data_to_label)
                self.label_to_data.append(data)
            trans_data.append((self.data_to_label[data]))
        self.classes_ = self.label_to_data
        return trans_data[:-1]  # unk_token 제외

    def transform(self, data_list):
        trans_data = []
        for data in data_list:
            trans_data.append(self.data_to_label.get(data, self.data_to_label[self.unk_token]))
        return trans_data

    def inverse_transform(self, data_list):
        inverse = []
        for data in data_list:
            inverse.append(self.label_to_data[data])
        return inverse
