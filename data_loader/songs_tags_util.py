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
    def __init__(self, dataset, song_meta, model_input_size, label_info):
        self.dataset = dataset
        self.song_meta = song_meta

        self.model_input_size = model_input_size
        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)

        self.splitter = ArenaSplitter()
        self.plylst_id_label_dict = self.get_plylst_id_label_dict(dataset)
        self.plylst_id_meta_label_dict = self.get_plylst_id_meta_label_dict(dataset, song_meta)


    def get_plylst_id_label_dict(self, dataset):
        plylst_id_label_dict = {}
        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            if not songs:
                continue
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_id_label_dict[each["id"]] = songs + tags
        return plylst_id_label_dict


    def get_plylst_id_meta_label_dict(self, dataset, song_meta):
        song_id_meta_info_dict = {}
        for each in song_meta:
            if each['id'] not in self.all_songs_set:
                continue
            song_id_meta_info_dict[each['id']] = each['artist_id_basket'] + each['song_gn_gnr_basket']

        plylst_id_meta_label_dict = {}
        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            if not songs:
                continue
            meta_label_set = set()
            for song in songs:
                meta_label_set.update(song_id_meta_info_dict[song])
            plylst_id_meta_label_dict[each["id"]] = list(meta_label_set)
        return plylst_id_meta_label_dict

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

            # song label 이 없는 경우 날라감
            label = self.plylst_id_label_dict.get(each["id"], [])
            if not label:
                continue

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

    def _mask(self, playlists, mask_cols, del_cols):
        q_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            for del_col in del_cols:
                q_pl[i][del_col] = []

            for col in mask_cols:
                mask_len = len(playlists[i][col])
                mask = np.full(mask_len, False)

                min_num = 1
                if mask_len < 2:
                    min_num = 0
                num = random.randint(min_num, mask_len//2)
                # mask[:mask_len // 2] = True
                mask[:num] = True
                np.random.shuffle(mask)

                q_pl[i][col] = list(np.array(q_pl[i][col])[mask])

        return q_pl

    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists[:int(tot * 0.3)]
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print('total: %d, Song only: %d, Song & Tags: %d, Tags only: %d, Title only: %d' % (
            len(playlists), len(song_only), len(song_and_tags), len(tags_only), len(title_only)))

        song_q = self._mask(song_only, ['songs'], ['tags'])
        songtag_q = self._mask(song_and_tags, ['songs', 'tags'], [])
        tag_q = self._mask(tags_only, ['tags'], ['songs'])
        title_q = self._mask(title_only, [], ['songs', 'tags'])

        q = song_q + songtag_q + tag_q + title_q

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])

        return q

    def make_dataset_v2(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        random_sample = self._mask_data(self.dataset)
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

    def make_pre_train_dataset(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue

            label = songs + tags
            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)


            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

    def make_dataset_v3(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

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
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result


    def make_dataset_v3_only_song(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs:
                continue


            # song label 이 없는 경우
            label = songs

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

    def make_dataset_v3_num(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if len(songs) <= 5:
                continue

            # song label 이 없는 경우
            label = self.plylst_id_label_dict.get(each["id"], [])
            if not label:
                continue

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

    def make_dataset_v4(self, shuffle=True):
        result = {"model_input": [], "A_length": [], 'label': [], 'meta_label': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            # song label 이 없는 경우
            label = self.plylst_id_label_dict.get(each["id"], [])
            if not label:
                continue
            meta_label = self.plylst_id_meta_label_dict.get(each["id"], [])
            if not meta_label:
                continue

            songs, tags = self.get_random_sampled_model_input(songs, tags)
            if not songs and not tags:
                continue


            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)
            meta_label = self.label_info.meta_label_encoder.transform(meta_label)

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)
            result["meta_label"].append(meta_label)

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
        self.pre_train_loss_check_dataset = self.make_pre_train_loss_check_dataset(question)

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

    def make_pre_train_loss_check_dataset(self, question):
        result = {"model_input": [], "A_length": [], 'label': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))

            if not songs and not tags:
                continue

            label = self.label_info.label_encoder.transform(songs+tags)

            model_input = convert_model_input(songs, tags)
            A_length = len(songs) + 2  # A_length: len(cls || songs || sep)

            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result["model_input"].append(pad_model_input)
            result["A_length"].append(A_length)
            result["label"].append(label)

        return result

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

