import random

import numpy as np
import util
from tqdm import tqdm

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


def make_train_val_set(model_input_output, input_bucket_size, output_bucket_size, label_info, sample=5, shuffle=False):
    bucket_dataset = {}

    for each in tqdm(model_input_output, total=len(model_input_output)):
        songs, tags, positive_label, negative_label = each

        for _ in range(sample):
            _negative_label = random.sample(negative_label, len(positive_label))
            songs, tags = get_random_sampled_model_input(songs, tags)
            model_input = convert_model_input(songs, tags)
            model_input_A_length = len(songs)+2  # A_length: len(cls || songs || sep)

            input_bucket = util.select_bucket(len(model_input), input_bucket_size)
            output_bucket = util.select_bucket(len(positive_label), output_bucket_size)

            bucket = (input_bucket, output_bucket)
            pad_model_input = label_info.label_encoder.transform(
                model_input + [label_info.pad_token] * (input_bucket - len(model_input)))
            pad_positive_label = label_info.label_encoder.transform(
                positive_label + [label_info.unk_token] * (output_bucket - len(positive_label)))
            pad_negative_label = label_info.label_encoder.transform(
                _negative_label + [label_info.unk_token] * (output_bucket - len(_negative_label)))

            if bucket not in bucket_dataset:
                bucket_dataset[bucket] = {}
                bucket_dataset[bucket]['model_input'] = []
                bucket_dataset[bucket]['model_input_A_length'] = []
                bucket_dataset[bucket]['positive_label'] = []
                bucket_dataset[bucket]['negative_label'] = []
            bucket_dataset[bucket]['model_input'].append(pad_model_input)
            bucket_dataset[bucket]['model_input_A_length'].append(model_input_A_length)
            bucket_dataset[bucket]['positive_label'].append(pad_positive_label)
            bucket_dataset[bucket]['negative_label'].append(pad_negative_label)

    for bucket in bucket_dataset:
        for key in bucket_dataset[bucket]:
            bucket_dataset[bucket][key] = np.array(bucket_dataset[bucket][key], np.int32)

        if shuffle:
            shuffle_index = np.array(range(len(bucket_dataset[bucket]['model_input'])))
            np.random.shuffle(shuffle_index)

            for key in bucket_dataset[bucket]:
                bucket_dataset[bucket][key] = bucket_dataset[bucket][key][shuffle_index]

    return bucket_dataset


def make_model_input_output(dataset, label_info):
    model_input_output = []

    all_songs_set = set(label_info.songs)
    all_tags_set = set(label_info.tags)

    for each in tqdm(dataset, total=len(dataset)):
        songs = list(filter(lambda song: song in all_songs_set, each['songs']))
        tags = list(filter(lambda tag: tag in all_tags_set, each['tags']))

        if not songs and not tags:
            continue

        positive_label = songs + tags

        negative_songs = util.get_k_negative_label(set(songs), label_info.songs, len(songs) * 3)
        negative_tags = util.get_k_negative_label(set(tags), label_info.tags, len(tags) * 3)
        negative_label = negative_songs + negative_tags

        model_input_output.append([songs, tags, positive_label, negative_label])

    return model_input_output

