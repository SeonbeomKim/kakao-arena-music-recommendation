import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tqdm import tqdm

import data_loader.plylst_title_util as plylst_title_util
import data_loader.songs_tags_util as songs_tags_util
import parameters
import util
from models.OrderlessBertAE import OrderlessBertAE
from models.TransformerEncoderAE import TransformerEncoderAE


class Ensemble:
    def __init__(self, label_info, song_issue_dict, sp, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # nvidia-smi의 k번째 gpu만 사용
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용

        self.plylst_id_seen_songs_dict = {}
        self.plylst_id_seen_tags_dict = {}
        self.plylst_id_plylst_updt_date_dict = {}
        self.song_issue_dict = song_issue_dict

        self.label_info = label_info
        self.sp = sp

        self.all_songs_set = set(self.label_info.songs)
        self.all_tags_set = set(self.label_info.tags)

        self.songs_tags_model_pad_idx = self.label_info.label_encoder.transform([self.label_info.pad_token])[0]
        self.plylst_title_model_pad_idx = self.sp.piece_to_id(self.label_info.pad_token)

    def load_plylst_title_model(self, plylst_title_saver_path, epoch_list=[]):
        model_sess_list = []
        for epoch in epoch_list:
            plylst_title_model_graph = tf.Graph()
            with plylst_title_model_graph.as_default():
                plylst_title_model = TransformerEncoderAE(
                    voca_size=len(self.sp),
                    embedding_size=parameters.embed_size,
                    is_embedding_scale=True,
                    max_sequence_length=parameters.title_max_sequence_length,
                    encoder_decoder_stack=parameters.stack,
                    multihead_num=parameters.multihead,
                    pad_idx=self.sp.piece_to_id(self.label_info.pad_token),
                    songs_num=len(self.label_info.songs),
                    tags_num=len(self.label_info.tags))

            plylst_title_model_sess = tf.Session(graph=plylst_title_model_graph, config=self.config)

            with plylst_title_model_sess.as_default():
                with plylst_title_model_graph.as_default():
                    tf.global_variables_initializer().run()
                    plylst_title_model.saver.restore(plylst_title_model_sess,
                                                     os.path.join(plylst_title_saver_path, '%d.ckpt' % epoch))
                    print('plylst_title_model restore:', os.path.join(plylst_title_saver_path, '%d.ckpt' % epoch))

            model_sess_list.append([plylst_title_model, plylst_title_model_sess])
        self.plylst_title_model = model_sess_list

    def load_songs_tags_model(self, songs_tags_saver_path, epoch_list=[]):
        model_sess_list = []
        for epoch in epoch_list:
            songs_tags_model_graph = tf.Graph()
            with songs_tags_model_graph.as_default():
                songs_tags_model = OrderlessBertAE(
                    voca_size=len(self.label_info.label_encoder.classes_),
                    embedding_size=parameters.embed_size,
                    is_embedding_scale=True,
                    max_sequence_length=parameters.max_sequence_length,
                    encoder_decoder_stack=parameters.stack,
                    multihead_num=parameters.multihead,
                    pad_idx=self.label_info.label_encoder.transform([self.label_info.pad_token])[0],
                    songs_num=len(self.label_info.songs),
                    tags_num=len(self.label_info.tags))

            songs_tags_model_sess = tf.Session(graph=songs_tags_model_graph, config=self.config)

            with songs_tags_model_sess.as_default():
                with songs_tags_model_graph.as_default():
                    tf.global_variables_initializer().run()
                    songs_tags_model.saver.restore(songs_tags_model_sess,
                                                   os.path.join(songs_tags_saver_path, '%d.ckpt' % epoch))
                    print('songs_tags_model restore:', os.path.join(songs_tags_saver_path, '%d.ckpt' % epoch))

            model_sess_list.append([songs_tags_model, songs_tags_model_sess])
        self.songs_tags_model = model_sess_list

    def songs_tags_do_reco(self, model_obj, model_input, plylst_id_list):
        plylst_id_reco_song_score_dict = {}
        plylst_id_reco_tag_score_dict = {}

        for each in model_obj:
            model = each[0]
            sess = each[1]

            reco_songs, reco_songs_score, reco_tags, reco_tags_score = sess.run(
                [model.reco_songs, model.reco_songs_score, model.reco_tags, model.reco_tags_score],
                {model.input_sequence_indices: model_input,
                 model.keep_prob: 1.0,
                 model.song_top_k: 30000,
                 model.tag_top_k: 300})

            for reco_song, reco_song_score, plylst_id in zip(reco_songs, reco_songs_score, plylst_id_list):
                reco_song = self.label_info.label_encoder.inverse_transform(reco_song)

                if plylst_id not in plylst_id_reco_song_score_dict:
                    plylst_id_reco_song_score_dict[plylst_id] = {}

                valid_data_num = 0
                for song, score in zip(reco_song, reco_song_score):
                    if valid_data_num >= 500:
                        break

                    if song in self.plylst_id_seen_songs_dict[plylst_id]:
                        continue
                    if self.song_issue_dict[song] > self.plylst_id_plylst_updt_date_dict[plylst_id]:
                        continue

                    valid_data_num += 1
                    if song not in plylst_id_reco_song_score_dict[plylst_id]:
                        plylst_id_reco_song_score_dict[plylst_id][song] = 0
                    plylst_id_reco_song_score_dict[plylst_id][song] += score

            for reco_tag, reco_tag_score, plylst_id in zip(reco_tags, reco_tags_score, plylst_id_list):
                reco_tag = self.label_info.label_encoder.inverse_transform(reco_tag)

                if plylst_id not in plylst_id_reco_tag_score_dict:
                    plylst_id_reco_tag_score_dict[plylst_id] = {}

                valid_data_num = 0
                for tag, score in zip(reco_tag, reco_tag_score):
                    if valid_data_num >= 50:
                        break

                    if tag in self.plylst_id_seen_tags_dict[plylst_id]:
                        continue

                    valid_data_num += 1
                    if tag not in plylst_id_reco_tag_score_dict[plylst_id]:
                        plylst_id_reco_tag_score_dict[plylst_id][tag] = 0
                    plylst_id_reco_tag_score_dict[plylst_id][tag] += score

        return plylst_id_reco_song_score_dict, plylst_id_reco_tag_score_dict

    def coldstart_do_reco(self, plylst_id):
        reco_songs = []
        reco_tags = self.label_info.tags[:10]
        for song in self.label_info.songs:  # top freq ?쒖꽌??
            if len(reco_songs) == 100:
                break
            if self.song_issue_dict[song] > self.plylst_id_plylst_updt_date_dict[plylst_id]:
                continue
            reco_songs.append(song)
        return reco_songs, reco_tags

    def do_reco(self, question_path, batch_size=128, title_song_weight=1.0, title_tag_weight=1.0):
        answers = []

        songs_tags_data = {'model_input': [], 'plylst_id_list': []}
        plylst_title_data = {'model_input': [], 'plylst_id_list': []}
        coldstart_plylst_id_list = []

        question = util.load_json(question_path)
        for each in tqdm(question, total=len(question), desc='Preprocess'):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            plylst_title = util.remove_special_char(each['plylst_title'])
            plylst_id = each['id']
            plylst_updt_date = each['updt_date']

            self.plylst_id_seen_songs_dict[plylst_id] = set(songs)
            self.plylst_id_seen_tags_dict[plylst_id] = set(tags)
            self.plylst_id_plylst_updt_date_dict[plylst_id] = util.convert_updt_date(plylst_updt_date)

            if songs or tags:
                model_input = songs_tags_util.convert_model_input(songs, tags, self.label_info.label_encoder)
                model_input += [self.songs_tags_model_pad_idx] * (parameters.max_sequence_length - len(model_input))
                songs_tags_data['model_input'].append(model_input)
                songs_tags_data['plylst_id_list'].append(plylst_id)

            if plylst_title:
                model_input = plylst_title_util.convert_model_input(plylst_title, self.label_info.cls_tokens[0],
                                                                    self.label_info.cls_tokens[1],
                                                                    self.label_info.sep_token,
                                                                    self.sp)
                model_input += [self.plylst_title_model_pad_idx] * (
                        parameters.title_max_sequence_length - len(model_input))
                plylst_title_data['model_input'].append(model_input)
                plylst_title_data['plylst_id_list'].append(plylst_id)

            if not songs and not tags and not plylst_title:
                coldstart_plylst_id_list.append(plylst_id)

        total_plylst_id_reco_song_score_dict = {}
        total_plylst_id_reco_tag_score_dict = {}

        # do songs_tags_model
        iter = int(np.ceil(len(songs_tags_data['model_input']) / batch_size))
        for i in tqdm(range(iter), desc='songs_tags_model'):
            plylst_id_reco_song_score_dict, plylst_id_reco_tag_score_dict = self.songs_tags_do_reco(
                self.songs_tags_model,
                model_input=songs_tags_data['model_input'][i * batch_size:(i + 1) * batch_size],
                plylst_id_list=songs_tags_data['plylst_id_list'][i * batch_size:(i + 1) * batch_size])

            for plylst_id in plylst_id_reco_song_score_dict:
                total_plylst_id_reco_song_score_dict[plylst_id] = plylst_id_reco_song_score_dict[plylst_id]
            for plylst_id in plylst_id_reco_tag_score_dict:
                total_plylst_id_reco_tag_score_dict[plylst_id] = plylst_id_reco_tag_score_dict[plylst_id]

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
                    total_plylst_id_reco_song_score_dict[plylst_id][song] += score * title_song_weight

            for plylst_id in plylst_id_reco_tag_score_dict:
                if plylst_id not in total_plylst_id_reco_tag_score_dict:
                    total_plylst_id_reco_tag_score_dict[plylst_id] = {}
                for tag, score in plylst_id_reco_tag_score_dict[plylst_id].items():
                    if tag not in total_plylst_id_reco_tag_score_dict[plylst_id]:
                        total_plylst_id_reco_tag_score_dict[plylst_id][tag] = 0
                    total_plylst_id_reco_tag_score_dict[plylst_id][tag] += score * title_tag_weight

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
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--question', required=True)
    args.add_argument('--fout', required=True)
    args.add_argument('--bs', type=int, default=128)
    args.add_argument('--gpu', type=int, default=6)
    config = args.parse_args()


    label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
    song_issue_dict = util.load(os.path.join(parameters.base_dir, parameters.song_issue_dict))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(parameters.base_dir, parameters.bpe_model_file))

    ensemble = Ensemble(label_info, song_issue_dict, sp, config.gpu)

    plylst_title_epoch_val_score_list = util.load(
        os.path.join(parameters.base_dir, 'plylst_title_epoch_val_score_list.pickle'))
    plylst_title_saver_path = plylst_title_epoch_val_score_list[-1]
    plylst_title_epoch_list = [plylst_title_epoch_val_score_list[0][0]]

    ensemble.load_plylst_title_model(
        plylst_title_saver_path ,
        epoch_list=plylst_title_epoch_list)


    songs_tags_epoch_val_score_list = util.load(
        os.path.join(parameters.base_dir, 'songs_tags_epoch_val_score_list.pickle'))
    songs_tags_saver_path = songs_tags_epoch_val_score_list[-1]
    songs_tags_epoch_list = list(map(lambda x:x[0], songs_tags_epoch_val_score_list[:2]))

    ensemble.load_songs_tags_model(
        songs_tags_saver_path,
        epoch_list=songs_tags_epoch_list)


    title_song_weight = 0.3
    title_tag_weight = 1.5
    question_path = config.question
    answers = ensemble.do_reco(question_path, batch_size=config.gpu, title_song_weight=title_song_weight, title_tag_weight=title_tag_weight)

    util.write_json(answers, config.fout)