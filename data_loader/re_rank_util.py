import random

import copy
import numpy as np
from tqdm import tqdm

import parameters
import util
from collections import Counter

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



class TrainReRankUtil:
    def __init__(self, dataset, song_meta, model_input_size, label_info):
        self.dataset = dataset
        self.song_meta = song_meta

        self.model_input_size = model_input_size
        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        self.splitter = ArenaSplitter()
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(dataset)
        self.song_id_meta_dict = self.get_song_id_meta_dict(song_meta)


    def get_plylst_id_label_dict(self, dataset):
        plylst_id_label_dict = {}
        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            if not songs:
                continue
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]] = songs + tags
        return plylst_id_label_dict


    def get_song_id_meta_dict(self, song_meta):
        song_id_meta_dict = {}
        for each in song_meta:
            if each['id'] not in self.all_songs_set:
                continue
            song_id_meta_dict[each['id']] = {}
            song_id_meta_dict[each['id']]['artist_id_basket'] = each['artist_id_basket']
            song_id_meta_dict[each['id']]['song_gn_gnr_basket'] = each['song_gn_gnr_basket']

        return song_id_meta_dict

    def get_random_sampled_model_input(self, songs, tags):
        # songs + tags는 최소 1개 보장됨.

        val_test_max_songs = 100  # val, test에는 최대 100개 노래 존재
        val_test_max_tags = 5  # val, test에는 최대 5개 태그 존재

        min_songs = 0
        if not tags:  # tag가 없으면 노래는 최소 1개 있어야함
            min_songs = 1
        valid_songs_num = min(len(songs), val_test_max_songs)
        songs = random.sample(songs, random.randint(min(min_songs, valid_songs_num), valid_songs_num))

        min_tags = 0
        if not songs:  # song이 없으면 tag는 최소 1개 있어야함
            min_tags = 1
        valid_tags_num = min(len(tags), val_test_max_tags)
        tags = random.sample(tags, random.randint(min(min_tags, valid_tags_num), valid_tags_num))

        return songs, tags


    def make_dataset(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs:
                continue

            label = songs + tags
            if not label:
                continue

            songs, _ = self.get_random_sampled_model_input(songs, tags)
            if not songs:
                continue

            artist = []
            genre = []
            for song in songs:
                artist.extend(self.song_id_meta_dict[song]['artist_id_basket'])
                genre.extend(self.song_id_meta_dict[song]['song_gn_gnr_basket'])
            artist = list(map(lambda x:x[0], Counter(artist).most_common(100)))
            genre = list(map(lambda x:x[0], Counter(genre).most_common(5)))

            model_input = convert_model_input(artist, genre)
            A_length = len(artist) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.meta_label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result



class ValReRankUtil(ArenaEvaluator):
    def __init__(self, question, answer, song_meta, reco_result, model_input_size, label_info):
        self.model_input_size = model_input_size

        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        # for loss check
        self.song_id_meta_dict = self.get_song_id_meta_dict(song_meta)
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(question, answer)
        self.loss_check_dataset = self.make_loss_check_dataset(question)

        # for ndcg check
        self.ndcg_check_dataset, self.answer_label = self.make_ndcg_check_dataset(reco_result, question, answer)

        self._idcgs = [self._idcg(i) for i in range(101)]
        super(ValReRankUtil, self).__init__()

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

            if not songs:
                continue

            label = self.label_info.label_encoder.transform(self.plylst_id_label_dict[each["id"]])

            artist = []
            genre = []
            for song in songs:
                artist.extend(self.song_id_meta_dict[song]['artist_id_basket'])
                genre.extend(self.song_id_meta_dict[song]['song_gn_gnr_basket'])
            artist = list(map(lambda x:x[0], Counter(artist).most_common(100)))
            genre = list(map(lambda x: x[0], Counter(genre).most_common(5)))

            model_input = convert_model_input(artist, genre)
            A_length = len(artist) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.meta_label_encoder.transform(
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


    def make_ndcg_check_dataset(self, reco_result, question, answer):
        result = {'model_input': [], 'A_length': [], 'id_list': [], 'reco_songs': [], 'reco_tags': []}
        answer_label = []

        id_reco_result_dict = {}
        for each in tqdm(reco_result, total=len(reco_result)):
            id_reco_result_dict[each["id"]] = {'songs':each['songs'], 'tags':each['tags']}


        id_answer_label_dict = {}
        for each in tqdm(answer, total=len(answer)):
            # songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            # tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            id_answer_label_dict[each["id"]] = {'songs':each['songs'], 'tags':each['tags']}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))

            if not songs:
                continue

            artist = []
            genre = []
            for song in songs:
                artist.extend(self.song_id_meta_dict[song]['artist_id_basket'])
                genre.extend(self.song_id_meta_dict[song]['song_gn_gnr_basket'])
            artist = list(map(lambda x:x[0], Counter(artist).most_common(100)))
            genre = list(map(lambda x:x[0], Counter(genre).most_common(5)))

            model_input = convert_model_input(artist, genre)
            A_length = len(artist) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.meta_label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            plylst_id = each["id"]
            answer_label.append(id_answer_label_dict[plylst_id])
            reco_songs = self.label_info.label_encoder.transform(id_reco_result_dict[plylst_id]['songs'])
            reco_tags = self.label_info.label_encoder.transform(id_reco_result_dict[plylst_id]['tags'])

            result['model_input'].append(pad_model_input)
            result['A_length'].append(A_length)
            result['id_list'].append(plylst_id)
            result['reco_songs'].append(reco_songs)
            result['reco_tags'].append(reco_tags)
        return result, answer_label

