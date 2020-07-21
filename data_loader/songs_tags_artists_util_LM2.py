import random

import numpy as np
import parameters
from tqdm import tqdm

import util

random.seed(888)
np.random.seed(888)


def get_artists(songs, song_artist_dict):
    artists_set = set()
    for song in songs:
        artists_set.update(song_artist_dict.get(song, []))
    return list(artists_set)


def make_mask_dataset(data, all_data_list, mask_token):
    # if len(data) <= 1:
    #     return [], [], []

    data = np.array(data)

    mask_position = np.random.randint(1, 101, size=len(data)) <= 15  # 1~15 즉 15%는 마스킹.
    mask_mode = mask_position.astype(np.int32)

    mask_method = np.random.randint(1, 11,
                                    size=sum(mask_position))  # 1~8: mask token, 9: change random token, 10: keep token

    mask_mode[mask_mode == 1] = mask_method
    mask_label = data[mask_position].tolist()

    # if not len(mask_label):
    #     return [], [], []

    masked_data = data.tolist()
    for idx, mode in enumerate(mask_mode):
        if mode == 0:
            continue

        if mode >= 1 and mode <= 8:
            masked_data[idx] = mask_token
        elif mode == 9:
            masked_data[idx] = random.choice(all_data_list)
        elif mode == 10:
            pass

    boolean_mask = mask_position.tolist()

    return masked_data, mask_label, boolean_mask


def convert_model_input(songs=[], tags=[], artists=[], label_encoder=None):
    result = ['@song_cls', '@tag_cls']
    if songs:
        result += songs
    if tags:
        result += tags
    if artists:
        result += artists
    if label_encoder:
        return label_encoder.transform(result)
    return result


class TrainUtil:
    def __init__(self, dataset, model_input_size, label_info):
        self.dataset = dataset

        self.model_input_size = model_input_size
        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)
        self.all_artists_set = set(label_info.artists)

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

    def make_dataset(self, shuffle=True):
        result = {"model_input": [], 'label': [], 'input_size': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = get_artists(songs, self.label_info.song_artist_dict)

            label = songs + tags + artists
            if not label:
                continue

            sampled_songs, sampled_tags = self.get_random_sampled_model_input(songs, tags)
            if not sampled_songs and not sampled_tags:
                continue
            sampled_artists = get_artists(sampled_songs, self.label_info.song_artist_dict)

            model_input = convert_model_input(sampled_songs, sampled_tags, sampled_artists)
            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            label = self.label_info.label_encoder.transform(label)

            result['input_size'].append(len(model_input))
            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        result['model_input'] = np.array(result['model_input'], dtype=np.int32)
        return result

    def make_pre_train_dataset(self, shuffle=True):
        result = {"model_input": [], 'input_size': [], 'mask_label': [], 'boolean_mask': []}

        if shuffle:
            random.shuffle(self.dataset)

        for each in tqdm(self.dataset, total=len(self.dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = get_artists(songs, self.label_info.song_artist_dict)

            label = songs + tags + artists
            if not label:
                continue

            sampled_songs, sampled_tags = self.get_random_sampled_model_input(songs, tags)
            if not sampled_songs and not sampled_tags:
                continue
            sampled_artists = get_artists(sampled_songs, self.label_info.song_artist_dict)

            songs_masked_data, mask_label, songs_boolean_mask = make_mask_dataset(sampled_songs,
                                                                                  self.label_info.songs,
                                                                                  self.label_info.sep_token)
            if not mask_label:
                continue
            mask_label = self.label_info.label_encoder.transform(mask_label)

            model_input = convert_model_input(songs_masked_data, sampled_tags, sampled_artists)
            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            boolean_mask = [False, False] + songs_boolean_mask
            pad_boolean_mask = boolean_mask + [False] * (self.model_input_size - len(boolean_mask))

            result['input_size'].append(len(model_input))
            result["model_input"].append(pad_model_input)
            result["mask_label"].append(mask_label)
            result["boolean_mask"].append(pad_boolean_mask)

        result['model_input'] = np.array(result['model_input'], dtype=np.int32)
        result['boolean_mask'] = np.array(result['boolean_mask'], dtype=np.bool)
        return result


class ValUtil:
    def __init__(self, question, answer, model_input_size, label_info):
        self.model_input_size = model_input_size

        self.label_info = label_info

        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)
        self.all_artists_set = set(label_info.artists)

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
        result = {"model_input": [], 'label': [], 'input_size': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = get_artists(songs, self.label_info.song_artist_dict)
            if not songs and not tags:
                continue

            answer_songs = list(filter(lambda song: song in self.all_songs_set,
                                       self.answer_plylst_id_songs_tags_dict[each['id']]['songs']))
            answer_tags = list(filter(lambda tag: tag in self.all_tags_set,
                                      self.answer_plylst_id_songs_tags_dict[each['id']]['tags']))
            answer_artists = get_artists(answer_songs, self.label_info.song_artist_dict)

            label = songs + answer_songs + tags + answer_tags + artists + answer_artists
            if not label:
                continue
            label = self.label_info.label_encoder.transform(label)

            model_input = convert_model_input(songs, tags, artists)
            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            result['input_size'].append(len(model_input))
            result["model_input"].append(pad_model_input)
            result["label"].append(label)

        result['model_input'] = np.array(result['model_input'], dtype=np.int32)
        return result

    def make_ndcg_check_dataset(self, question):
        result = {'model_input': [], 'id_list': [], 'input_size': [], 'seen_songs_set': [], 'seen_tags_set': [],
                  'plylst_updt_date': [], 'gt': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = get_artists(songs, self.label_info.song_artist_dict)

            if not songs and not tags and not artists:
                continue

            model_input = convert_model_input(songs, tags, artists)
            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))

            gt = self.answer_plylst_id_songs_tags_dict[each["id"]]
            gt['id'] = each["id"]

            result['gt'].append(gt)
            result['model_input'].append(pad_model_input)
            result['input_size'].append(len(model_input))
            result['id_list'].append(each["id"])
            result['seen_songs_set'].append(set(songs))
            result['seen_tags_set'].append(set(tags))
            result['plylst_updt_date'].append(util.convert_updt_date(each["updt_date"]))

        result['model_input'] = np.array(result['model_input'], dtype=np.int32)
        return result

    def make_pre_train_accuracy_check_dataset(self, question):
        result = {"model_input": [], 'input_size': [], 'mask_label': [], 'boolean_mask': []}

        for each in tqdm(question, total=len(question)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = get_artists(songs, self.label_info.song_artist_dict)
            if not songs and not tags:
                continue

            songs_masked_data, mask_label, songs_boolean_mask = make_mask_dataset(songs,
                                                                                  self.label_info.songs,
                                                                                  self.label_info.sep_token)
            if not mask_label:
                continue
            mask_label = self.label_info.label_encoder.transform(mask_label)

            model_input = convert_model_input(songs_masked_data, tags, artists)
            pad_model_input = self.label_info.label_encoder.transform(
                model_input + [self.label_info.pad_token] * (self.model_input_size - len(model_input)))
            boolean_mask = [False, False] + songs_boolean_mask
            pad_boolean_mask = boolean_mask + [False] * (self.model_input_size - len(boolean_mask))

            result['input_size'].append(len(model_input))
            result["model_input"].append(pad_model_input)
            result["mask_label"].append(mask_label)
            result["boolean_mask"].append(pad_boolean_mask)

        result['model_input'] = np.array(result['model_input'], dtype=np.int32)
        result['boolean_mask'] = np.array(result['boolean_mask'], dtype=np.bool)
        return result
