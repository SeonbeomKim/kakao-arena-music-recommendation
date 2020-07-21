import os
import random

import numpy as np
import parameters
import sentencepiece as spm
from tqdm import tqdm

import util

random.seed(777)
np.random.seed(777)


def dump_plylst_title(dataset, fout):
    with open(fout, 'w', encoding='utf-8', errors='ignore') as o:
        for each in dataset:
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue
            o.write(plylst_title + '\n')


def train_sentencepiece(dataset):
    dump_plylst_title(dataset, os.path.join(parameters.base_dir, parameters.plylst_titles))

    # sentencepiece
    spm.SentencePieceTrainer.train(
        input=os.path.join(parameters.base_dir, parameters.plylst_titles),
        model_prefix=parameters.bpe_model_prefix,
        vocab_size=parameters.bpe_voca_size,
        character_coverage=parameters.bpe_character_coverage,
        model_type='bpe',
        user_defined_symbols=[parameters.songs_cls_token, parameters.tags_cls_token, parameters.artists_cls_token,
                              parameters.sep_token, parameters.mask_token, parameters.pad_token])


def convert_model_input(name, sentencepiece, model_input_size):
    result = [sentencepiece.piece_to_id(parameters.songs_cls_token)] + [sentencepiece.piece_to_id(
        parameters.tags_cls_token)] + [sentencepiece.piece_to_id(parameters.artists_cls_token)] + sentencepiece.encode(
        name)[:model_input_size - 3] + [sentencepiece.piece_to_id(parameters.sep_token)]
    return result


def make_mask_dataset(converted_model_input, sentencepiece):
    prefix = converted_model_input[:3]
    postfix = converted_model_input[-1:]
    title_part = np.array(converted_model_input[3:-1])

    if len(title_part) <= 1:
        return [], [], []

    mask_position = np.random.randint(1, 101, size=len(title_part)) <= 15  # 1~15 즉 15%는 마스킹.
    mask_mode = mask_position.astype(np.int32)

    mask_method = np.random.randint(1, 11,
                                    size=sum(mask_position))  # 1~8: mask token, 9: change random token, 10: keep token

    mask_mode[mask_mode == 1] = mask_method
    mask_label = title_part[mask_position]

    if not len(mask_label):
        return [], [], []

    for idx, mode in enumerate(mask_mode):
        if mode == 0:
            continue

        if mode >= 1 and mode <= 8:
            title_part[idx] = sentencepiece.piece_to_id(parameters.mask_token)
        elif mode == 9:
            title_part[idx] = random.randint(0, len(sentencepiece) - 1)
        elif mode == 10:
            pass

    model_input = prefix + title_part.tolist() + postfix
    boolean_mask = [False] * len(prefix) + mask_position.tolist() + [False] * len(postfix)

    return model_input, mask_label.tolist(), boolean_mask


class TrainUtil:
    def __init__(self, dataset, model_input_size, label_info, sentencepiece):
        self.dataset = dataset
        self.model_input_size = model_input_size
        self.label_info = label_info
        self.sentencepiece = sentencepiece

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

    def make_dataset(self, shuffle=True):
        result = {"model_input": [], 'label': []}
        pad_idx = self.sentencepiece.piece_to_id(parameters.pad_token)

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = util.get_artists(songs, self.label_info.song_artist_dict)

            label = songs + tags + artists

            model_input = convert_model_input(plylst_title, self.sentencepiece, self.model_input_size)
            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        return result

    def make_pre_train_dataset(self, shuffle=True):
        result = {"model_input": [], 'mask_label': [], 'boolean_mask': []}
        pad_idx = self.sentencepiece.piece_to_id(parameters.pad_token)

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            model_input = convert_model_input(plylst_title, self.sentencepiece, self.model_input_size)
            model_input, mask_label, boolean_mask = make_mask_dataset(model_input, self.sentencepiece)
            if not model_input:
                continue

            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))
            pad_boolean_mask = boolean_mask + [False] * (self.model_input_size - len(boolean_mask))

            result["model_input"].append(pad_model_input)
            result["mask_label"].append(mask_label)
            result["boolean_mask"].append(pad_boolean_mask)

        return result


class ValUtil:
    def __init__(self, question, answer, model_input_size, label_info, sentencepiece):
        self.model_input_size = model_input_size

        self.label_info = label_info
        self.sentencepiece = sentencepiece

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        self.answer_plylst_id_songs_tags_dict = self.get_plylst_id_songs_tags_dict(answer)

        # for loss check
        self.loss_check_dataset = self.make_loss_check_dataset(question)

        # for ndcg check
        self.ndcg_check_dataset = self.make_ndcg_check_dataset(question)

        # for pretrain accuracy
        self.pre_train_accuracy_check_dataset = self.make_pre_train_accuracy_check_dataset(question)

    def get_plylst_id_songs_tags_dict(self, data):
        plylst_id_label_dict = {}
        for each in data:
            plylst_id_label_dict[each["id"]] = {'songs': each['songs'], 'tags': each['tags']}
        return plylst_id_label_dict

    def make_loss_check_dataset(self, question):
        dataset = {"model_input": [], 'label': []}
        pad_idx = self.sentencepiece.piece_to_id(parameters.pad_token)

        for each in tqdm(question, total=len(question)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = util.get_artists(songs, self.label_info.song_artist_dict)

            answer_songs = list(filter(lambda song: song in self.all_songs_set,
                                       self.answer_plylst_id_songs_tags_dict[each['id']]['songs']))
            answer_tags = list(filter(lambda tag: tag in self.all_tags_set,
                                      self.answer_plylst_id_songs_tags_dict[each['id']]['tags']))
            answer_artists = util.get_artists(answer_songs, self.label_info.song_artist_dict)

            label = songs + answer_songs + tags + answer_tags + artists + answer_artists
            if not label:
                continue
            label = self.label_info.label_encoder.transform(label)

            model_input = convert_model_input(plylst_title, self.sentencepiece, self.model_input_size)
            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))

            dataset["model_input"].append(pad_model_input)
            dataset["label"].append(label)

        return dataset

    def make_ndcg_check_dataset(self, question):
        result = {'model_input': [], 'id_list': [], 'seen_songs_set': [], 'seen_tags_set': [],
                  'plylst_updt_date': [], 'gt': []}

        pad_idx = self.sentencepiece.piece_to_id(parameters.pad_token)

        for each in tqdm(question, total=len(question)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            model_input = convert_model_input(plylst_title, self.sentencepiece, self.model_input_size)

            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))

            gt = self.answer_plylst_id_songs_tags_dict[each["id"]]
            gt['id'] = each["id"]

            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            result['gt'].append(gt)
            result['model_input'].append(pad_model_input)
            result['id_list'].append(each["id"])
            result['seen_songs_set'].append(set(songs))
            result['seen_tags_set'].append(set(tags))
            result['plylst_updt_date'].append(util.convert_updt_date(each["updt_date"]))

        return result

    def make_pre_train_accuracy_check_dataset(self, question):
        result = {"model_input": [], 'mask_label': [], 'boolean_mask': []}
        pad_idx = self.sentencepiece.piece_to_id(parameters.pad_token)

        for each in tqdm(question, total=len(question)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            model_input = convert_model_input(plylst_title, self.sentencepiece, self.model_input_size)
            model_input, mask_label, boolean_mask = make_mask_dataset(model_input, self.sentencepiece)
            if not model_input:
                continue

            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))
            pad_boolean_mask = boolean_mask + [False] * (self.model_input_size - len(boolean_mask))

            result["model_input"].append(pad_model_input)
            result["mask_label"].append(mask_label)
            result["boolean_mask"].append(pad_boolean_mask)

        return result
