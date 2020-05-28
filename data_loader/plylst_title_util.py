import random

import numpy as np
from tqdm import tqdm

import util

random.seed(777)
np.random.seed(777)

def dump_plylst_title(dataset, fout):
    with open(fout, 'w', encoding='utf-8', errors='ignore') as o:
        for each in dataset:
            plylst_title = each['plylst_title']
            if not plylst_title:
                continue
            o.write(plylst_title + '\n')


def convert_model_input(name, cls_token, sep_token, sentencepiece, enable_sampling=False, alpha=0.2):
    result = [sentencepiece.piece_to_id(cls_token)] + sentencepiece.encode(name, enable_sampling=enable_sampling,
                                                                           alpha=alpha) + [
                 sentencepiece.piece_to_id(sep_token)]
    return result


def make_train_val_set(model_input_output, input_bucket_size, output_bucket_size, label_info, sentencepiece, sample=5,
                       shuffle=False):
    bucket_dataset = {}

    pad_idx = sentencepiece.piece_to_id(label_info.pad_token)

    for each in tqdm(model_input_output, total=len(model_input_output)):
        plylst_title, positive_label, negative_label = each

        for _ in range(sample):
            _negative_label = random.sample(negative_label, len(positive_label))
            model_input = convert_model_input(plylst_title, label_info.cls_token, label_info.sep_token, sentencepiece,
                                              enable_sampling=True, alpha=0.1)

            input_bucket = util.select_bucket(len(model_input), input_bucket_size)
            output_bucket = util.select_bucket(len(positive_label), output_bucket_size)

            bucket = (input_bucket, output_bucket)
            pad_model_input = model_input + [pad_idx] * (input_bucket - len(model_input))
            pad_positive_label = label_info.label_encoder.transform(
                positive_label + [label_info.unk_token] * (output_bucket - len(positive_label)))
            pad_negative_label = label_info.label_encoder.transform(
                _negative_label + [label_info.unk_token] * (output_bucket - len(_negative_label)))

            if bucket not in bucket_dataset:
                bucket_dataset[bucket] = {}
                bucket_dataset[bucket]['model_input'] = []
                bucket_dataset[bucket]['positive_label'] = []
                bucket_dataset[bucket]['negative_label'] = []
            bucket_dataset[bucket]['model_input'].append(pad_model_input)
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
        plylst_title = each['plylst_title']

        if not plylst_title:
            continue

        if not songs and not tags:
            continue

        positive_label = songs + tags

        negative_songs = util.get_k_negative_label(set(songs), label_info.songs, len(songs) * 3)
        negative_tags = util.get_k_negative_label(set(tags), label_info.tags, len(tags) * 3)
        negative_label = negative_songs + negative_tags

        model_input_output.append([plylst_title, positive_label, negative_label])

    return model_input_output
