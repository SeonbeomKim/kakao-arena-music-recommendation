import argparse
import os

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from models.TransformerEncoderAE import TransformerEncoderAE
from tqdm import tqdm

import data_loader.plylst_title_util as plylst_title_util
import data_loader.songs_tags_util as songs_tags_util
import parameters
import util
from models.OrderlessBertAE import OrderlessBertAE

args = argparse.ArgumentParser()
args.add_argument('--input_path')
args.add_argument('--output_path')
config = args.parse_args()
input_path = config.input_path
output_path = config.output_path


class EnsemblePredictor:
    def __init__(self, plylst_title_saver_path, songs_tags_saver_path, label_info, sp):
        self.plylst_title_saver_path = plylst_title_saver_path
        self.songs_tags_saver_path = songs_tags_saver_path
        self.label_info = label_info
        self.sp = sp
        self.all_songs_set = set(label_info.songs)
        self.all_tags_set = set(label_info.tags)
        self.plylst_title_model, self.plylst_title_model_sess, \
        self.songs_tags_model, self.songs_tags_model_sess = self.model_load(plylst_title_saver_path,
                                                                            songs_tags_saver_path)

    def model_load(self, plylst_title_saver_path, songs_tags_saver_path):
        plylst_title_model_graph = tf.Graph()
        with plylst_title_model_graph.as_default():
            plylst_title_model = TransformerEncoderAE(
                voca_size=len(self.sp),
                songs_tags_size=len(self.label_info.label_encoder.classes_),
                embedding_size=parameters.embed_size,
                is_embedding_scale=True,
                max_sequence_length=parameters.title_max_sequence_length,
                encoder_decoder_stack=parameters.stack,
                multihead_num=parameters.multihead,
                pad_idx=sp.piece_to_id(self.label_info.pad_token),
                unk_idx=self.label_info.label_encoder.transform([self.label_info.unk_token])[0],
                songs_num=len(self.label_info.songs),
                tags_num=len(self.label_info.tags))

        songs_tags_model_graph = tf.Graph()
        with songs_tags_model_graph.as_default():
            songs_tags_model = OrderlessBertAE(
                voca_size=len(self.label_info.label_encoder.classes_),
                embedding_size=parameters.embed_size,  # 128인경우 850MB 먹음.
                is_embedding_scale=True,
                max_sequence_length=parameters.max_sequence_length,
                encoder_decoder_stack=parameters.stack,
                multihead_num=parameters.multihead,
                pad_idx=self.label_info.label_encoder.transform([self.label_info.pad_token])[0],
                unk_idx=self.label_info.label_encoder.transform([self.label_info.unk_token])[0],
                songs_num=len(self.label_info.songs),
                tags_num=len(self.label_info.tags))

        # gpu 할당 및 session 생성
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # nvidia-smi의 k번째 gpu만 사용
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용

        plylst_title_model_sess = tf.Session(graph=plylst_title_model_graph, config=config)
        songs_tags_model_sess = tf.Session(graph=songs_tags_model_graph, config=config)

        with plylst_title_model_sess.as_default():
            with plylst_title_model_graph.as_default():
                tf.global_variables_initializer().run()
                plylst_title_model.saver.restore(plylst_title_model_sess, plylst_title_saver_path)
                print('plylst_title_model restore:', plylst_title_restore)

        with songs_tags_model_sess.as_default():
            with songs_tags_model_graph.as_default():
                tf.global_variables_initializer().run()
                songs_tags_model.saver.restore(songs_tags_model_sess, songs_tags_saver_path)
                print('songs_tags_model restore:', songs_tags_restore)
        return plylst_title_model, plylst_title_model_sess, songs_tags_model, songs_tags_model_sess

    def plylst_title_do_reco(self, model_input, song_top_k=100, tag_top_k=50):
        model = self.plylst_title_model
        sess = self.plylst_title_model_sess

        reco_songs, reco_songs_score, reco_tags, reco_tags_score = sess.run(
            [model.reco_songs, model.reco_songs_score, model.reco_tags, model.reco_tags_score],
            {model.input_sequence_indices: model_input,
             model.keep_prob: 1.0,
             model.song_top_k: song_top_k,
             model.tag_top_k: tag_top_k})
        return list(zip(reco_songs, reco_songs_score)), list(zip(reco_tags, reco_tags_score))

    def songs_tags_do_reco(self, model_input, A_length, song_top_k=100, tag_top_k=50):
        model = self.songs_tags_model
        sess = self.songs_tags_model_sess

        reco_songs, reco_songs_score, reco_tags, reco_tags_score = sess.run(
            [model.reco_songs, model.reco_songs_score, model.reco_tags, model.reco_tags_score],
            {model.input_sequence_indices: model_input,
             model.A_length: A_length,
             model.keep_prob: 1.0,
             model.song_top_k: song_top_k,
             model.tag_top_k: tag_top_k})
        return list(zip(reco_songs, reco_songs_score)), list(zip(reco_tags, reco_tags_score))

    def merge_dict(self, dict_A, dict_B, id, weight=1):
        if id not in dict_A:
            dict_A[id] = {}
        for key, value in dict_B.items():
            if key not in dict_A[id]:
                dict_A[id][key] = 0
            dict_A[id][key] += value * weight

    def predict(self, dataset, song_meta, pred_songs_num=100, pred_tags_num=10, plylst_title_song_weight=1.,
                plylst_title_tag_weight=1., batch_size=64):
        answers = []

        sp = self.sp
        label_info = self.label_info

        song_issue_dict = {}
        for each in song_meta:
            song_issue_dict[each["id"]] = int(each['issue_date'])

        id_seen_songs_dict = {}
        id_seen_tags_dict = {}
        id_plylst_updt_date_dict = {}
        songs_tags_model_info = {'id_list': [], 'model_input': [], 'A_length': []}
        plylst_title_model_info = {'id_list': [], 'model_input': []}

        no_data = 0
        total_id_reco_songs = {}
        total_id_reco_tags = {}
        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            seen_songs = set(songs)
            seen_tags = set(tags)
            plylst_title = each['plylst_title']
            plylst_updt_date = util.convert_updt_date(each["updt_date"])
            plylst_id = each["id"]

            if not songs and not tags and not plylst_title:
                no_data += 1
                # print('No songs No tags No plylst_title')
                cold_start_reco_songs = []
                cold_start_reco_tags = label_info.tags[:pred_tags_num]

                for song in label_info.songs:  # top freq 순서임
                    if len(cold_start_reco_songs) == pred_songs_num:
                        break
                    if song_issue_dict[song] > plylst_updt_date:
                        continue
                    cold_start_reco_songs.append(song)

                answers.append({
                    "id": plylst_id,
                    "songs": cold_start_reco_songs,
                    "tags": cold_start_reco_tags,
                })
                continue

            if len(songs) + len(tags):
                model_input = songs_tags_util.convert_model_input(songs, tags, label_info.label_encoder)
                A_length = len(songs) + 2
                songs_tags_model_info['id_list'].append(plylst_id)
                songs_tags_model_info['model_input'].append(model_input)
                songs_tags_model_info['A_length'].append(A_length)


            if plylst_title:
                model_input = plylst_title_util.convert_model_input(
                    plylst_title, label_info.cls_token, label_info.sep_token, sp)
                plylst_title_model_info['id_list'].append(plylst_id)
                plylst_title_model_info['model_input'].append(model_input)

            id_seen_songs_dict[plylst_id] = seen_songs
            id_seen_tags_dict[plylst_id] = seen_tags
            id_plylst_updt_date_dict[plylst_id] = plylst_updt_date


        print('# no_data: %d' % no_data)
        # songs_tags_model run
        data_num = len(songs_tags_model_info['id_list'])
        epoch = int(np.ceil(data_num / batch_size))
        for i in tqdm(range(epoch)):
            batch_model_input = util.fill_na(
                songs_tags_model_info['model_input'][i * batch_size:(i + 1) * batch_size],
                label_info.label_encoder.transform([label_info.pad_token])[0]).astype(np.int32)
            batch_A_length = songs_tags_model_info['A_length'][i * batch_size:(i + 1) * batch_size]
            batch_id_list = songs_tags_model_info['id_list'][i * batch_size:(i + 1) * batch_size]

            reco_songs, reco_tags = self.songs_tags_do_reco(batch_model_input, batch_A_length,
                                                            song_top_k=1000,
                                                            tag_top_k=100)
            for (_reco_songs, _reco_score), _id in zip(reco_songs, batch_id_list):
                self.merge_dict(total_id_reco_songs, dict(zip(_reco_songs, _reco_score)), _id, weight=1.)
            for (_reco_tags, _reco_score), _id in zip(reco_tags, batch_id_list):
                self.merge_dict(total_id_reco_tags, dict(zip(_reco_tags, _reco_score)), _id, weight=1.)
        del songs_tags_model_info

        # plylst title model run
        data_num = len(plylst_title_model_info['id_list'])
        epoch = int(np.ceil(data_num / batch_size))
        for i in tqdm(range(epoch)):
            batch_model_input = util.fill_na(
                plylst_title_model_info['model_input'][i * batch_size:(i + 1) * batch_size],
                sp.piece_to_id(label_info.pad_token)).astype(np.int32)
            batch_id_list = plylst_title_model_info['id_list'][i * batch_size:(i + 1) * batch_size]

            reco_songs, reco_tags = self.plylst_title_do_reco(batch_model_input,
                                                              song_top_k=1000,
                                                              tag_top_k=100)
            for (_reco_songs, _reco_score), _id in zip(reco_songs, batch_id_list):
                self.merge_dict(total_id_reco_songs, dict(zip(_reco_songs, _reco_score)), _id,
                                weight=plylst_title_song_weight)
            for (_reco_tags, _reco_score), _id in zip(reco_tags, batch_id_list):
                self.merge_dict(total_id_reco_tags, dict(zip(_reco_tags, _reco_score)), _id,
                                weight=plylst_title_tag_weight)

        # 이미 담겨있는 노래, 태그, 플레이리스트 업데이트 날짜보다 발매일 늦은 노래 제외 및 정렬하고 자르기
        for _id in total_id_reco_songs:
            # 노래
            seen_songs = id_seen_songs_dict[_id]
            plylst_updt_date = id_plylst_updt_date_dict[_id]
            filtered_songs = []
            for song, score in total_id_reco_songs[_id].items():
                if len(filtered_songs) == pred_songs_num:
                    break
                if song in seen_songs:
                    continue
                if song_issue_dict[song] > plylst_updt_date:
                    continue
                filtered_songs.append([song, score])
            total_id_reco_songs[_id] = list(
                map(lambda x: x[0], sorted(filtered_songs, key=lambda x: x[1], reverse=True)))[:pred_songs_num]

            # 태그
            seen_tags = id_seen_tags_dict[_id]
            filtered_tags = list(
                filter(lambda tag_score: tag_score[0] not in seen_tags, total_id_reco_tags[_id].items()))
            total_id_reco_tags[_id] = list(
                map(lambda x: x[0], sorted(filtered_tags, key=lambda x: x[1], reverse=True)))[:pred_tags_num]

            answers.append({
                "id": _id,
                "songs": label_info.label_encoder.inverse_transform(total_id_reco_songs[_id]),
                "tags": label_info.label_encoder.inverse_transform(total_id_reco_tags[_id]),
            })
        return answers


if __name__ == "__main__":
    gpu = 4

    label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(parameters.base_dir, parameters.bpe_model_file))

    plylst_title_restore = 70
    songs_tags_restore = 145
    plylst_title_saver_path = 'saver_decay_new_titleAE_emb128_stack4_head4_lr_0.00010_tags_loss_weight_0.55_negative_loss_weight_0.55/%d.ckpt' % (
        plylst_title_restore)
    songs_tags_saver_path = 'saver_decay_new_AE_batch_emb128_stack4_head4_lr_0.00050_tags_loss_weight_0.55_negative_loss_weight_0.55/%d.ckpt' % (
        songs_tags_restore)

    # plylst_title_restore = 75
    # songs_tags_restore = 90
    # plylst_title_saver_path = 'saver_decay_new_titleAE_emb128_stack4_head4_lr_0.00010_tags_loss_weight_0.55_negative_loss_weight_0.55/%d.ckpt' % (
    #     plylst_title_restore)
    # songs_tags_saver_path = 'saver_decay_new_AE_batch_emb128_stack4_head4_lr_0.00080_tags_loss_weight_0.55_negative_loss_weight_0.55/%d.ckpt' % (
    #     songs_tags_restore)

    print(plylst_title_saver_path)
    print(songs_tags_saver_path)

    # predictor
    self = EnsemblePredictor(plylst_title_saver_path, songs_tags_saver_path, label_info, sp)

    # val_set = util.load_json('dataset/questions/val.json')
    val_set = util.load_json(input_path)

    song_meta = util.load_json('dataset/song_meta.json')

    plylst_title_song_weight = 0.8
    plylst_title_tag_weight = 1.5

    print(input_path)
    print(os.path.join(output_path,
                       "results_%0.3f_%0.3f.json" % (plylst_title_song_weight, plylst_title_tag_weight)))

    reco_result = self.predict(val_set, song_meta, pred_songs_num=100, pred_tags_num=10,
                               plylst_title_song_weight=plylst_title_song_weight,
                               plylst_title_tag_weight=plylst_title_tag_weight)
    util.write_json(reco_result,
                    os.path.join(output_path,
                                 "results_%0.3f_%0.3f.json" % (plylst_title_song_weight, plylst_title_tag_weight)))
