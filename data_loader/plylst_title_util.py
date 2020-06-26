import os
import random

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

import parameters
import util
from evaluate import ArenaEvaluator

random.seed(888)
np.random.seed(888)


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
        user_defined_symbols=['@song_cls', '@tag_cls', '@sep', '@mask', '@pad', '@unk'])

    # 아래 path에 저장됨 os.path.join(parameters.base_dir, parameters.bpe_model_file)


def convert_model_input(name, song_cls_token, tag_cls_token, sep_token, sentencepiece, enable_sampling=False,
                        alpha=0.2):
    result = [sentencepiece.piece_to_id(song_cls_token)] + [sentencepiece.piece_to_id(
        tag_cls_token)] + sentencepiece.encode(name, enable_sampling=enable_sampling, alpha=alpha) + [
                 sentencepiece.piece_to_id(sep_token)]
    return result


class TrainPlylstTitleUtil:
    def __init__(self, dataset, model_input_size, label_info, sentencepiece):
        self.dataset = dataset
        self.model_input_size = model_input_size
        self.label_info = label_info
        self.sentencepiece = sentencepiece

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

    def make_dataset(self, shuffle=True):
        result = {"model_input": [], 'label': []}
        pad_idx = self.sentencepiece.piece_to_id(self.label_info.pad_token)

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            label = songs + tags

            model_input = convert_model_input(plylst_title, self.label_info.cls_tokens[0],
                                              self.label_info.cls_tokens[1], self.label_info.sep_token,
                                              self.sentencepiece, enable_sampling=True, alpha=0.1)
            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        return result


class ValPlylstTitleUtil(ArenaEvaluator):
    def __init__(self, question, answer, song_meta, model_input_size, label_info, sentencepiece):
        self.model_input_size = model_input_size

        self.label_info = label_info
        self.sentencepiece = sentencepiece

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        # for loss check
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(question, answer)
        self.loss_check_dataset = self.make_loss_check_dataset(question)

        # for ndcg check
        self.plylst_id_seen_songs_dict = {}
        self.plylst_id_seen_tags_dict = {}
        self.plylst_id_plylst_updt_date_dict = {}
        self.ndcg_check_dataset, self.answer_label = self.make_ndcg_check_dataset(question, answer)

        super(ValPlylstTitleUtil, self).__init__()

    def get_plylst_id_label_dict(self, question, answer):
        plylst_id_label_dict = {}
        for each in question:
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]] = songs + tags

        for each in answer:
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]].extend(songs + tags)
        return plylst_id_label_dict

    def make_loss_check_dataset(self, question):
        dataset = {"model_input": [], 'label': []}
        pad_idx = self.sentencepiece.piece_to_id(self.label_info.pad_token)

        for each in tqdm(question, total=len(question)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            model_input = convert_model_input(plylst_title, self.label_info.cls_tokens[0],
                                              self.label_info.cls_tokens[1], self.label_info.sep_token,
                                              self.sentencepiece)
            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))
            label = self.label_info.label_encoder.transform(self.plylst_id_label_dict[each["id"]])

            dataset["model_input"].append(pad_model_input)
            dataset["label"].append(label)

        return dataset

    def _eval(self, rec_list):
        music_ndcg = 0.0
        tag_ndcg = 0.0

        for gt, rec in zip(self.answer_label, rec_list):
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_list)
        tag_ndcg = tag_ndcg / len(rec_list)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def make_ndcg_check_dataset(self, question, answer):
        result = {'model_input': [], 'id_list': []}
        pad_idx = self.sentencepiece.piece_to_id(self.label_info.pad_token)

        answer_label = []

        id_answer_label_dict = {}
        for each in tqdm(answer, total=len(answer)):
            # songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            # tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            id_answer_label_dict[each["id"]] = {'songs': each['songs'], 'tags': each['tags']}

        for each in tqdm(question, total=len(question)):
            plylst_title = util.remove_special_char(each['plylst_title'])
            if not plylst_title:
                continue

            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            plylst_id = each["id"]
            self.plylst_id_seen_songs_dict[plylst_id] = set(songs)
            self.plylst_id_seen_tags_dict[plylst_id] = set(tags)
            self.plylst_id_plylst_updt_date_dict[plylst_id] = util.convert_updt_date(each["updt_date"])

            answer_label.append(id_answer_label_dict[plylst_id])

            model_input = convert_model_input(plylst_title, self.label_info.cls_tokens[0],
                                              self.label_info.cls_tokens[1], self.label_info.sep_token,
                                              self.sentencepiece)

            pad_model_input = model_input + [pad_idx] * (self.model_input_size - len(model_input))

            result['model_input'].append(pad_model_input)
            result['id_list'].append(plylst_id)

        return result, answer_label
