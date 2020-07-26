import os

import argparse
import data_loader.plylst_title_util as plylst_title_util
import data_loader.songs_tags_artists_util as songs_tags_artists_util
import numpy as np
import parameters
import sentencepiece as spm
import tensorflow as tf
from models.OrderlessBertAE import OrderlessBertAE
from models.TitleBert import TitleBert
from tqdm import tqdm

import util

args = argparse.ArgumentParser()
args.add_argument('--bs', type=int, default=128)
args.add_argument('--gpu', type=int, default=6)
args.add_argument('--title_importance', type=float, default=0.85)
args.add_argument('--title_tag_weight', type=float, default=0.8)
args.add_argument('--question_path', type=str, default='./dataset/val.json')
args.add_argument('--out_path', type=str, default='./reco_result/results.json')

config = args.parse_args()
bs = config.bs
gpu = config.gpu
title_importance = config.title_importance
title_tag_weight = config.title_tag_weight
question_path = config.question_path
out_path = config.out_path


class Ensemble:
    def __init__(self, label_info, song_issue_dict, sp, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.plylst_id_seen_songs_dict = {}
        self.plylst_id_seen_tags_dict = {}
        self.plylst_id_plylst_updt_date_dict = {}
        self.song_issue_dict = song_issue_dict

        self.label_info = label_info
        self.sp = sp

        self.all_songs_set = set(self.label_info.songs)
        self.all_tags_set = set(self.label_info.tags)

        self.songs_tags_artists_model_pad_idx = self.label_info.label_encoder.transform([parameters.pad_token])[0]
        self.plylst_title_model_pad_idx = self.sp.piece_to_id(parameters.pad_token)

    def load_plylst_title_model(self, plylst_title_saver_path, epoch):
        plylst_title_model_graph = tf.Graph()
        with plylst_title_model_graph.as_default():
            plylst_title_model = TitleBert(
                voca_size=len(sp),
                embedding_size=parameters.title_model_embed_size,
                is_embedding_scale=True,
                max_sequence_length=parameters.title_model_max_sequence_length,
                encoder_decoder_stack=parameters.title_model_stack,
                multihead_num=parameters.title_model_multihead,
                pad_idx=sp.piece_to_id(parameters.pad_token),
                songs_num=len(label_info.songs),
                tags_num=len(label_info.tags))

        plylst_title_model_sess = tf.Session(graph=plylst_title_model_graph, config=self.config)

        with plylst_title_model_sess.as_default():
            with plylst_title_model_graph.as_default():
                tf.global_variables_initializer().run()
                plylst_title_model.saver.restore(plylst_title_model_sess,
                                                 os.path.join(plylst_title_saver_path, '%d.ckpt' % epoch))
                print('plylst_title_model restore:', os.path.join(plylst_title_saver_path, '%d.ckpt' % epoch))
        self.plylst_title_model = [plylst_title_model, plylst_title_model_sess]

    def load_songs_tags_artists_model(self, songs_tags_artists_saver_path, epoch):
        songs_tags_artists_model_graph = tf.Graph()
        with songs_tags_artists_model_graph.as_default():
            songs_tags_artists_model = OrderlessBertAE(
                voca_size=len(label_info.label_encoder.classes_),
                embedding_size=parameters.songs_tags_artists_model_embed_size,
                is_embedding_scale=True,
                encoder_decoder_stack=parameters.songs_tags_artists_model_stack,
                multihead_num=parameters.songs_tags_artists_model_multihead,
                pad_idx=label_info.label_encoder.transform([parameters.pad_token])[0],
                songs_num=len(label_info.songs),
                tags_num=len(label_info.tags),
                artists_num=len(label_info.artists))

        songs_tags_artists_model_sess = tf.Session(graph=songs_tags_artists_model_graph, config=self.config)

        with songs_tags_artists_model_sess.as_default():
            with songs_tags_artists_model_graph.as_default():
                tf.global_variables_initializer().run()
                songs_tags_artists_model.saver.restore(songs_tags_artists_model_sess,
                                                       os.path.join(songs_tags_artists_saver_path, '%d.ckpt' % epoch))
                print('songs_tags_artists_model restore:',
                      os.path.join(songs_tags_artists_saver_path, '%d.ckpt' % epoch))
        self.songs_tags_artists_model = [songs_tags_artists_model, songs_tags_artists_model_sess]

    def songs_tags_do_reco(self, model_obj, model_input, plylst_id_list, song_N=500, tag_N=50):
        plylst_id_reco_song_score_dict = {}
        plylst_id_reco_tag_score_dict = {}

        model = model_obj[0]
        sess = model_obj[1]

        reco_songs, reco_songs_score, reco_tags, reco_tags_score = sess.run(
            [model.reco_songs, model.reco_songs_score, model.reco_tags, model.reco_tags_score],
            {model.input_sequence_indices: model_input,
             model.keep_prob: 1.0,
             model.song_top_k: 20000,  # 필터링되는 경우 있어서 충분히 많이 추출
             model.tag_top_k: 300})

        for reco_song, reco_song_score, plylst_id in zip(reco_songs, reco_songs_score, plylst_id_list):
            reco_song = self.label_info.label_encoder.inverse_transform(reco_song)

            plylst_id_reco_song_score_dict[plylst_id] = {}
            for song, score in zip(reco_song, reco_song_score):
                if len(plylst_id_reco_song_score_dict[plylst_id]) == song_N:
                    break

                if song in self.plylst_id_seen_songs_dict[plylst_id]:
                    continue
                if self.song_issue_dict[song] > self.plylst_id_plylst_updt_date_dict[plylst_id]:
                    continue

                plylst_id_reco_song_score_dict[plylst_id][song] = score

        for reco_tag, reco_tag_score, plylst_id in zip(reco_tags, reco_tags_score, plylst_id_list):
            reco_tag = self.label_info.label_encoder.inverse_transform(reco_tag)

            plylst_id_reco_tag_score_dict[plylst_id] = {}
            for tag, score in zip(reco_tag, reco_tag_score):
                if len(plylst_id_reco_tag_score_dict[plylst_id]) == tag_N:
                    break

                if tag in self.plylst_id_seen_tags_dict[plylst_id]:
                    continue

                plylst_id_reco_tag_score_dict[plylst_id][tag] = score

        return plylst_id_reco_song_score_dict, plylst_id_reco_tag_score_dict

    def coldstart_do_reco(self, plylst_id):
        reco_songs = []
        reco_tags = self.label_info.tags[:10]
        for song in self.label_info.songs:
            if len(reco_songs) == 100:
                break
            if self.song_issue_dict[song] > self.plylst_id_plylst_updt_date_dict[plylst_id]:
                continue
            reco_songs.append(song)
        return reco_songs, reco_tags

    def do_reco(self, question_path, batch_size=128, title_importance=0.85, title_tag_weight=0.8):
        answers = []

        songs_tags_artists_data = {'model_input': [], 'plylst_id_list': []}
        plylst_title_data = {'model_input': [], 'plylst_id_list': []}
        coldstart_plylst_id_list = []

        plylst_id_songs_tags_num = {}

        question = util.load_json(question_path)
        for each in tqdm(question, total=len(question), desc='Preprocess'):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            artists = util.get_artists(songs, self.label_info.song_artist_dict)
            plylst_title = util.remove_special_char(each['plylst_title'])
            plylst_id = each['id']
            plylst_updt_date = each['updt_date']

            self.plylst_id_seen_songs_dict[plylst_id] = set(songs)
            self.plylst_id_seen_tags_dict[plylst_id] = set(tags)
            self.plylst_id_plylst_updt_date_dict[plylst_id] = util.convert_updt_date(plylst_updt_date)

            plylst_id_songs_tags_num[plylst_id] = len(songs + tags)
            if songs or tags:
                model_input = songs_tags_artists_util.convert_model_input(songs, tags, artists,
                                                                          self.label_info.label_encoder)
                model_input += [self.songs_tags_artists_model_pad_idx] * (
                        parameters.songs_tags_artists_model_max_sequence_length - len(model_input))
                songs_tags_artists_data['model_input'].append(model_input)
                songs_tags_artists_data['plylst_id_list'].append(plylst_id)

            if plylst_title:
                model_input = plylst_title_util.convert_model_input(plylst_title, self.sp,
                                                                    parameters.title_model_max_sequence_length)
                model_input += [self.plylst_title_model_pad_idx] * (
                        parameters.title_model_max_sequence_length - len(model_input))
                plylst_title_data['model_input'].append(model_input)
                plylst_title_data['plylst_id_list'].append(plylst_id)

            if not songs and not tags and not plylst_title:
                coldstart_plylst_id_list.append(plylst_id)

        total_plylst_id_reco_song_score_dict = {}
        total_plylst_id_reco_tag_score_dict = {}

        # do songs_tags_artists_model
        iter = int(np.ceil(len(songs_tags_artists_data['model_input']) / batch_size))
        for i in tqdm(range(iter), desc='songs_tags_artists_model'):
            plylst_id_reco_song_score_dict, plylst_id_reco_tag_score_dict = self.songs_tags_do_reco(
                self.songs_tags_artists_model,
                model_input=songs_tags_artists_data['model_input'][i * batch_size:(i + 1) * batch_size],
                plylst_id_list=songs_tags_artists_data['plylst_id_list'][i * batch_size:(i + 1) * batch_size])

            for plylst_id in plylst_id_reco_song_score_dict:
                if plylst_id not in total_plylst_id_reco_song_score_dict:
                    total_plylst_id_reco_song_score_dict[plylst_id] = {}
                for song, score in plylst_id_reco_song_score_dict[plylst_id].items():
                    if song not in total_plylst_id_reco_song_score_dict[plylst_id]:
                        total_plylst_id_reco_song_score_dict[plylst_id][song] = 0
                    total_plylst_id_reco_song_score_dict[plylst_id][song] += score * plylst_id_songs_tags_num[
                        plylst_id] / (title_importance + plylst_id_songs_tags_num[plylst_id])

            for plylst_id in plylst_id_reco_tag_score_dict:
                if plylst_id not in total_plylst_id_reco_tag_score_dict:
                    total_plylst_id_reco_tag_score_dict[plylst_id] = {}
                for tag, score in plylst_id_reco_tag_score_dict[plylst_id].items():
                    if tag not in total_plylst_id_reco_tag_score_dict[plylst_id]:
                        total_plylst_id_reco_tag_score_dict[plylst_id][tag] = 0
                    total_plylst_id_reco_tag_score_dict[plylst_id][tag] += score

        # do plylst_title_model
        iter = int(np.ceil(len(plylst_title_data['model_input']) / batch_size))
        for i in tqdm(range(iter), desc='plylst_title_model'):
            plylst_id_reco_song_score_dict, plylst_id_reco_tag_score_dict = self.songs_tags_do_reco(
                self.plylst_title_model,
                model_input=plylst_title_data['model_input'][i * batch_size:(i + 1) * batch_size],
                plylst_id_list=plylst_title_data['plylst_id_list'][i * batch_size:(i + 1) * batch_size])

            for plylst_id in plylst_id_reco_song_score_dict:
                if plylst_id not in total_plylst_id_reco_song_score_dict:
                    total_plylst_id_reco_song_score_dict[plylst_id] = {}
                for song, score in plylst_id_reco_song_score_dict[plylst_id].items():
                    if song not in total_plylst_id_reco_song_score_dict[plylst_id]:
                        total_plylst_id_reco_song_score_dict[plylst_id][song] = 0
                    total_plylst_id_reco_song_score_dict[plylst_id][
                        song] += score * title_importance / (title_importance + plylst_id_songs_tags_num[plylst_id])

            for plylst_id in plylst_id_reco_tag_score_dict:
                if plylst_id not in total_plylst_id_reco_tag_score_dict:
                    total_plylst_id_reco_tag_score_dict[plylst_id] = {}
                for tag, score in plylst_id_reco_tag_score_dict[plylst_id].items():
                    if tag not in total_plylst_id_reco_tag_score_dict[plylst_id]:
                        total_plylst_id_reco_tag_score_dict[plylst_id][tag] = 0
                    total_plylst_id_reco_tag_score_dict[plylst_id][
                        tag] += score * title_tag_weight

        # 두개 모델 종합해서 추천
        for plylst_id in total_plylst_id_reco_song_score_dict:
            reco_songs = list(map(lambda x: x[0], sorted(list(total_plylst_id_reco_song_score_dict[plylst_id].items()),
                                                         key=lambda x: x[1], reverse=True)[:100]))
            reco_tags = list(map(lambda x: x[0], sorted(list(total_plylst_id_reco_tag_score_dict[plylst_id].items()),
                                                        key=lambda x: x[1], reverse=True)[:10]))
            answers.append({
                "id": plylst_id,
                "songs": reco_songs,
                "tags": reco_tags,
            })

        # cold_start
        for plylst_id in tqdm(coldstart_plylst_id_list, total=len(coldstart_plylst_id_list), desc='coldstart_reco'):
            reco_songs, reco_tags = self.coldstart_do_reco(plylst_id)
            answers.append({
                "id": plylst_id,
                "songs": reco_songs,
                "tags": reco_tags,
            })

        return answers


if __name__ == "__main__":
    label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
    song_issue_dict = util.load(os.path.join(parameters.base_dir, parameters.song_issue_dict))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(parameters.base_dir, parameters.bpe_model_file))

    ensemble = Ensemble(label_info, song_issue_dict, sp, gpu)

    plylst_title_best_model_dict = util.load('plylst_title_best_model_dict.pickle')
    plylst_title_model_best_epoch = plylst_title_best_model_dict['epoch']
    plylst_title_model_saver_path = plylst_title_best_model_dict['saver_path']

    ensemble.load_plylst_title_model(
        plylst_title_model_saver_path,
        epoch=plylst_title_model_best_epoch)

    songs_tags_artists_best_model_dict = util.load('songs_tags_artists_best_model_dict.pickle')
    songs_tags_artists_model_best_epoch = songs_tags_artists_best_model_dict['epoch']
    songs_tags_artists_model_saver_path = songs_tags_artists_best_model_dict['saver_path']

    ensemble.load_songs_tags_artists_model(
        songs_tags_artists_model_saver_path,
        epoch=songs_tags_artists_model_best_epoch)

    # 추천
    answers = ensemble.do_reco(
        question_path=question_path, batch_size=bs, title_importance=title_importance,
        title_tag_weight=title_tag_weight)
    # 저장
    util.write_json(answers, out_path)
