import random

import numpy as np
from tqdm import tqdm

import parameters
import util

from split_data import ArenaSplitter

random.seed(777)
np.random.seed(777)


def convert_model_input(songs, tags, label_encoder=None):
    result = ['@cls']
    if songs:
        result += songs
    result += ['@sep']
    if tags:
        result += tags
    result += ['@sep']

    if label_encoder:
        return label_encoder.transform(result)
    return result


def get_random_sample(data, min_sample_size, max_sample_size):
    if not data:
        return []
    return random.sample(data, random.randint(min_sample_size, max_sample_size))


def get_random_sample2(data, min_dropout=0.4, max_dropout=1):
    if not data:
        return []

    keep_ratio = (1 - random.randint(min_dropout * 100, max_dropout * 100 - 1) / 100)
    keep_num = max(int(len(data) * keep_ratio), 1)
    return random.sample(data, keep_num)



def get_random_sampled_model_input(songs, tags):
    # songs + tags는 최소 1개 보장됨.

    val_test_max_songs = 100  # val, test에는 최대 100개 노래 존재
    val_test_max_tags = 5  # val, test에는 최대 5개 태그 존재

    min_songs = 0
    if not tags:  # tag가 없으면 노래는 최소 1개 있어야함
        min_songs = 1
    valid_songs_num = min(len(songs), val_test_max_songs)
    songs = random.sample(songs, random.randint(min_songs, valid_songs_num))

    min_tags = 0
    if not songs:  # song이 없으면 tag는 최소 1개 있어야함
        min_tags = 1
    valid_tags_num = min(len(tags), val_test_max_tags)
    tags = random.sample(tags, random.randint(min_tags, valid_tags_num))

    return songs, tags


def make_train_val_set(dataset, model_input_size, label_info, sample=5, shuffle=False):
    result = {"model_input":[], "A_length":[], 'label':[]}

    splitter = ArenaSplitter()

    all_songs_set = set(label_info.songs)
    all_tags_set = set(label_info.tags)

    plylst_id_label_dict = {}
    for i in tqdm(range(len(dataset))):
        plylst_id = dataset[i]["id"]
        # plylst_title = each['plylst_title']
        dataset[i]['songs'] = list(filter(lambda song: song in all_songs_set, dataset[i]['songs']))
        dataset[i]['tags'] = list(filter(lambda tag: tag in all_tags_set, dataset[i]['tags']))

        plylst_id_label_dict[plylst_id] = dataset[i]['songs'] + dataset[i]['tags']


    for _ in range(sample):
        if shuffle:
            random.shuffle(dataset)

        random_sample, _ = splitter._mask_data(dataset)
        for each in tqdm(random_sample, total=len(random_sample)):
            plylst_id = each["id"]
            songs = each['songs']
            tags = each['tags']

            if not songs and not tags:
                continue

            label = plylst_id_label_dict[plylst_id]

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = label_info.label_encoder.transform(
                model_input + [label_info.pad_token] * (model_input_size - len(model_input)))
            label = label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

    return result


def make_model_input_output(dataset, label_info, negative_sample_size=1000):
    model_input_output = []

    all_songs_set = set(label_info.songs)
    all_tags_set = set(label_info.tags)

    for each in tqdm(dataset, total=len(dataset)):
        songs = list(filter(lambda song: song in all_songs_set, each['songs']))
        tags = list(filter(lambda tag: tag in all_tags_set, each['tags']))

        if not songs and not tags:
            continue

        model_input_output.append([songs, tags])
    return model_input_output
