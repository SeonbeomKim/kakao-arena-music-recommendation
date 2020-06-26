import random

import numpy as np
from tqdm import tqdm

import parameters
import util
from evaluate import ArenaEvaluator
from split_data import ArenaSplitter

random.seed(888)
np.random.seed(888)


def convert_model_input(songs, tags, label_encoder=None):
    result = ['@song_cls', '@tag_cls']
    if songs:
        result += songs
    if tags:
        result += tags

    if label_encoder:
        return label_encoder.transform(result)
    return result


class TrainSongsTagsUtil:
    def __init__(self, dataset, song_meta, model_input_size, label_info):
        self.dataset = dataset
        self.song_meta = song_meta

        self.model_input_size = model_input_size
        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        self.splitter = ArenaSplitter()


    def get_random_sampled_model_input(self, songs, tags):
        min_songs = 0
        if not tags:  # tag가 없으면 노래는 최소 1개 있어야함
            min_songs = 1
        valid_songs_num = min(len(songs), parameters.val_test_max_songs)
        songs = random.sample(songs, random.randint(min(min_songs, valid_songs_num), valid_songs_num))

        min_tags = 0
        if not songs:  # song이 없으면 tag는 최소 1개 있어야함
            min_tags = 1
        valid_tags_num = min(len(tags), parameters.val_test_max_tags)
        tags = random.sample(tags, random.randint(min(min_tags, valid_tags_num), valid_tags_num))

        return songs, tags


    def make_dataset_v3(self, shuffle=True):
        result = {"model_input": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            label = songs + tags
            if not label:
                continue

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue

            model_input = convert_model_input(songs, tags)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        return result

    # TODO 데이터마다 song 1개 tag 0개, song 0개, tag 1개인 경우에 대해서 전부 학습에 사용하기
    def make_dataset_v4(self, shuffle=True):
        result = {"model_input": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            label = songs + tags
            if not label:
                continue

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue

            model_input = convert_model_input(songs, tags)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        return result

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
        self.plylst_id_answer_dict = self.get_plylst_id_answer_dict(answer)
        self.song_id_meta_dict = self.get_song_id_meta_dict(song_meta)
        self.plylst_id_seen_songs_dict = {}
        self.plylst_id_seen_tags_dict = {}
        self.plylst_id_plylst_updt_date_dict = {}
        self.ndcg_check_dataset, self.answer_label = self.make_ndcg_check_dataset(question)

        super(ValSongsTagsUtil, self).__init__()


    def get_song_id_meta_dict(self, song_meta):
        song_id_meta_dict = {}
        for each in song_meta:
            if each['id'] not in self.all_songs_set:
                continue
            song_id_meta_dict[each['id']] = {}
            song_id_meta_dict[each['id']]['artist_id_basket'] = each['artist_id_basket']
            song_id_meta_dict[each['id']]['song_gn_gnr_basket'] = each['song_gn_gnr_basket']

        return song_id_meta_dict

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

    def get_plylst_id_answer_dict(self, answer):
        plylst_id_answer_dict = {}
        for each in tqdm(answer, total=len(answer)):
            plylst_id_answer_dict[each["id"]] = {'songs': each['songs'], 'tags': each['tags']}
        return plylst_id_answer_dict


    def make_loss_check_dataset(self, question):
        result = {"model_input": [], 'label': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            label = self.label_info.label_encoder.transform(self.plylst_id_label_dict[each["id"]])

            model_input = convert_model_input(songs, tags)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result["model_input"].append(pad_model_input)
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

    def make_ndcg_check_dataset(self, question):
        result = {'model_input': [], 'id_list': []}
        answer_label = []

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            plylst_id = each["id"]
            self.plylst_id_seen_songs_dict[plylst_id] = set(songs)
            self.plylst_id_seen_tags_dict[plylst_id] = set(tags)
            self.plylst_id_plylst_updt_date_dict[plylst_id] = util.convert_updt_date(each["updt_date"])

            answer_label.append(self.plylst_id_answer_dict[plylst_id])

            model_input = convert_model_input(songs, tags)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result['model_input'].append(pad_model_input)
            result['id_list'].append(plylst_id)
        return result, answer_label
