import random

import copy
import numpy as np
from tqdm import tqdm

import parameters
import util

from split_data import ArenaSplitter
from evaluate import ArenaEvaluator

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


class TrainSongsTagsUtil:
    def __init__(self, dataset, model_input_size, label_info):
        self.dataset = dataset
        self.model_input_size = model_input_size
        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        self.splitter = ArenaSplitter()
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(dataset)


    def get_plylst_id_label_dict(self, dataset):
        plylst_id_label_dict = {}
        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]] = songs + tags
        return plylst_id_label_dict


    def make_dataset(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        random_sample, _ = self.splitter._mask_data(self.dataset)
        for each in tqdm(random_sample, total=len(random_sample)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            label = self.plylst_id_label_dict[each["id"]]

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result
    #
    # def make_dataset(self, shuffle=False):
    #     result = []
    #
    #     for each in tqdm(self.dataset, total=len(self.dataset)):
    #         songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
    #         tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
    #         label = self.label_info.label_encoder.transform(songs + tags)
    #
    #         sample_songs = [random.sample(songs, i) for i in range(len(songs) // 2 + 1)]
    #         sample_tags = [random.sample(tags, i) for i in range(len(tags) // 2 + 1)]
    #
    #         random.shuffle(sample_songs)
    #         random.shuffle(sample_tags)
    #
    #         for sample_song in sample_songs[:len(sample_songs) // 5 + 1]:
    #             for sample_tag in sample_tags[:3]:
    #                 if not sample_song and not sample_tag:
    #                     continue
    #
    #                 model_input = convert_model_input(sample_song, sample_tag)
    #                 A_length = len(sample_song) + 2  # A_length: len(cls || songs || sep)
    #
    #                 pad_model_input = self.label_info.label_encoder.transform(
    #                     model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
    #
    #                 result.append((pad_model_input, A_length, label))
    #
    #     if shuffle:
    #         random.shuffle(result)
    #
    #     result_dict = {"model_input": [], "A_length": [], 'label': []}
    #     for each in tqdm(result, total=len(result)):
    #         result_dict["model_input"].append(each[0])
    #         result_dict["A_length"].append(each[1])
    #         result_dict["label"].append(each[2])
    #
    #     return result_dict


class ValSongsTagsUtil(ArenaEvaluator):
    def __init__(self, question, answer, song_meta, model_input_size, label_info):
        self.model_input_size = model_input_size

        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        # for loss check
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(question, answer)
        self.loss_check_dataset = self.make_loss_check_dataset(question)

        # for ndcg check
        self.id_seen_songs_dict = {}
        self.id_seen_tags_dict = {}
        self.id_plylst_updt_date_dict = {}
        self.song_issue_dict = {}
        for each in song_meta:
            self.song_issue_dict[each["id"]] = int(each['issue_date'])
        self.ndcg_check_dataset, self.answer_label = self.make_ndcg_check_dataset(question, answer)

        self._idcgs = [self._idcg(i) for i in range(101)]
        super(ValSongsTagsUtil, self).__init__()


    def get_plylst_id_label_dict(self, question, answer):
        plylst_id_label_dict = {}
        for each in question:
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]] = songs+tags

        for each in answer:
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]].extend(songs+tags)
        return plylst_id_label_dict


    def make_loss_check_dataset(self, question):
        result = {"model_input": [], "A_length": [], 'label': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            label = self.label_info.label_encoder.transform(self.plylst_id_label_dict[each["id"]])

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

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
        result = {'model_input': [], 'A_length': [], 'id_list': []}
        answer_label = []


        id_answer_label_dict = {}
        for each in tqdm(answer, total=len(answer)):
            # songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            # tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            id_answer_label_dict[each["id"]] = {'songs':each['songs'], 'tags':each['tags']}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            plylst_id = each["id"]
            self.id_seen_songs_dict[plylst_id] = set(self.label_info.label_encoder.transform(songs))
            self.id_seen_tags_dict[plylst_id] = set(self.label_info.label_encoder.transform(tags))
            self.id_plylst_updt_date_dict[plylst_id] = util.convert_updt_date(each["updt_date"])

            answer_label.append(id_answer_label_dict[plylst_id])

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result['model_input'].append(pad_model_input)
            result['A_length'].append(A_length)
            result['id_list'].append(plylst_id)

        return result, answer_label

