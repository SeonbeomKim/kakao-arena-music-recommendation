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





def make_train_val_set(model_input_output, model_input_size, label_info, sentencepiece, sample=5, shuffle=False):
    dataset = {"model_input":[], 'label':[]}
    pad_idx = sentencepiece.piece_to_id(label_info.pad_token)

    for _ in range(sample):
        if shuffle:
            random.shuffle(model_input_output)

        for each in tqdm(model_input_output, total=len(model_input_output)):
            plylst_title, songs, tags = each
            label = songs+tags

            model_input = convert_model_input(plylst_title, label_info.cls_token, label_info.sep_token, sentencepiece,
                                              enable_sampling=True, alpha=0.1)
            pad_model_input = model_input + [pad_idx] * (model_input_size - len(model_input))
            label = label_info.label_encoder.transform(label)

            dataset["model_input"].append(pad_model_input)
            dataset["label"].append(label)

    return dataset




def make_model_input_output(dataset, label_info, negative_sample_size=1000):
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
        #
        # negative_songs = util.get_k_negative_label(set(songs), label_info.songs, negative_sample_size)
        # negative_tags = util.get_k_negative_label(set(tags), label_info.tags, negative_sample_size)
        # negative_label = negative_songs + negative_tags

        model_input_output.append([plylst_title, songs, tags])
        # model_input_output.append([plylst_title, positive_label, negative_label])

    return model_input_output
