import os

import sentencepiece as spm
import tensorflow as tf
from tqdm import tqdm

import data_loader.plylst_title_util as plylst_title_util
import data_loader.songs_tags_util as songs_tags_util
import parameters
import util
from models.OrderlessBert import OrderlessBert
from models.TransformerEncoder import TransformerEncoder


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
            plylst_title_model = TransformerEncoder(
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
            songs_tags_model = OrderlessBert(
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
            {model.input_sequence_indices: [model_input],
             model.keep_prob: 1.0,
             model.song_top_k: song_top_k,
             model.tag_top_k: tag_top_k})
        return list(zip(reco_songs[0], reco_songs_score[0])), list(zip(reco_tags[0], reco_tags_score[0]))

    def songs_tags_do_reco(self, model_input, A_length, song_top_k=100, tag_top_k=50):
        model = self.songs_tags_model
        sess = self.songs_tags_model_sess

        reco_songs, reco_songs_score, reco_tags, reco_tags_score = sess.run(
            [model.reco_songs, model.reco_songs_score, model.reco_tags, model.reco_tags_score],
            {model.input_sequence_indices: [model_input],
             model.A_length: [A_length],
             model.keep_prob: 1.0,
             model.song_top_k: song_top_k,
             model.tag_top_k: tag_top_k})
        return list(zip(reco_songs[0], reco_songs_score[0])), list(zip(reco_tags[0], reco_tags_score[0]))

    def predict(self, dataset, pred_songs_num=100, pred_tags_num=10, plylst_title_weight=1.):
        answers = []

        sp = self.sp
        label_info = self.label_info

        for each in tqdm(dataset, total=len(dataset)):
            songs = list(filter(lambda song: song in self.all_songs_set, each['songs']))
            tags = list(filter(lambda tag: tag in self.all_tags_set, each['tags']))
            seen_songs = set(each['songs'])
            seen_tags = set(each['tags'])
            plylst_title = each['plylst_title']

            if not songs and not tags and not plylst_title:
                print('No songs No tags No plylst_title')
                continue

            total_reco_songs = {}
            total_reco_tags = {}
            if len(songs) + len(tags):
                model_input = songs_tags_util.convert_model_input(songs, tags, label_info.label_encoder)
                A_length = len(songs) + 2

                reco_songs, reco_tags = self.songs_tags_do_reco(model_input, A_length,
                                                                song_top_k=pred_songs_num + len(songs),
                                                                tag_top_k=pred_tags_num + len(tags))

                total_reco_songs.update(dict(reco_songs))
                total_reco_tags.update(dict(reco_tags))

            if plylst_title:
                model_input = plylst_title_util.convert_model_input(
                    plylst_title, label_info.cls_token, label_info.sep_token, sp)

                reco_songs, reco_tags = self.plylst_title_do_reco(model_input,
                                                                  song_top_k=pred_songs_num + len(songs),
                                                                  tag_top_k=pred_tags_num + len(tags))
                reco_songs = dict(reco_songs)
                reco_tags = dict(reco_tags)

                for reco_song in reco_songs:
                    score = reco_songs[reco_song]
                    if reco_song not in total_reco_songs:
                        total_reco_songs[reco_song] = 0
                    total_reco_songs[reco_song] += score * plylst_title_weight

                for reco_tag in reco_tags:
                    score = reco_tags[reco_tag]
                    if reco_tag not in total_reco_tags:
                        total_reco_tags[reco_tag] = 0
                    total_reco_tags[reco_tag] += score * plylst_title_weight

            # 이미 담겨있는 노래, 태그 제외
            for reco_song in list(total_reco_songs.keys()):
                if reco_song in seen_songs:
                    total_reco_songs.pop(reco_song)
            for reco_tag in list(total_reco_tags.keys()):
                if reco_tag in seen_tags:
                    total_reco_tags.pop(reco_tag)

            reco_songs = list(
                map(lambda x: x[0], sorted(list(total_reco_songs.items()), key=lambda x: x[1], reverse=True)))[
                         :pred_songs_num]
            reco_tags = list(
                map(lambda x: x[0], sorted(list(total_reco_tags.items()), key=lambda x: x[1], reverse=True)))[
                        :pred_tags_num]

            answers.append({
                "id": each["id"],
                "songs": label_info.label_encoder.inverse_transform(reco_songs),
                "tags": label_info.label_encoder.inverse_transform(reco_tags),
            })
        return answers


if __name__ == "__main__":
    gpu = 3
    song_loss_weight = 1.

    plylst_title_lr = 0.00005  # config.lr
    plylst_title_restore = 18  # config.restore

    songs_tags_lr = 0.00005  # config.lr
    songs_tags_restore = 8  # config.restore

    label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(parameters.base_dir, parameters.bpe_model_file))

    plylst_title_restore = 26
    songs_tags_restore = 12
    plylst_title_saver_path = './saver_title_emb%d_stack%d_head%d_lr_%0.5f_song_loss_weight_%0.2f/%d.ckpt' % (
        parameters.embed_size, parameters.stack, parameters.multihead, plylst_title_lr, song_loss_weight,
        plylst_title_restore)
    songs_tags_saver_path = './saver_AE_emb%d_stack%d_head%d_lr_%0.5f_song_loss_weight_%0.2f/%d.ckpt' % (
        parameters.embed_size, parameters.stack, parameters.multihead, songs_tags_lr, song_loss_weight,
        songs_tags_restore)

    # predictor
    predictor = EnsemblePredictor(plylst_title_saver_path, songs_tags_saver_path, label_info, sp)

    val_set = util.load_json('dataset/questions/val.json')

    plylst_title_weight = [1.0, 0.8, 0.7, 0.5, 0.3, 0.1]

    for weight in plylst_title_weight:
        reco_result = predictor.predict(val_set, pred_songs_num=100, pred_tags_num=10, plylst_title_weight=weight)
        util.write_json(reco_result, "results/results_%0.2f.json" % weight)
