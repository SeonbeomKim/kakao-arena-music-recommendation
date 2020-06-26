import argparse
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import data_loader.songs_tags_util as songs_tags_util
import parameters
import util
from models.OrderlessBertAE import OrderlessBertAE

args = argparse.ArgumentParser()
args.add_argument('--bs', type=int, default=128)
args.add_argument('--gpu', type=int, default=6)
args.add_argument('--tags_loss_weight', type=float, default=0.15)
args.add_argument('--negative_loss_weight', type=float, default=1.0)
args.add_argument('--warmup_steps', type=float, default=4000)
config = args.parse_args()
bs = config.bs
gpu = config.gpu
tags_loss_weight = config.tags_loss_weight
negative_loss_weight = config.negative_loss_weight
warmup_steps = config.warmup_steps


def get_lr(step_num):
    '''
    https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf
    step_num(training_steps):  number of iterations, ie. the number of times the optimizer update was run
        This number also equals the number of mini batches that were processed.
    '''
    lr = (parameters.embed_size ** -0.5) * min((step_num ** -0.5), (step_num * (warmup_steps ** -1.5)))
    return lr


def train(model, train_util, iter, batch_size=64, keep_prob=0.9, tags_loss_weight=0.5, negative_loss_weight=0.5):
    model_train_dataset = train_util.make_dataset_v3(shuffle=True)
    loss = 0

    data_num = len(model_train_dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        step_num = ((iter - 1) * epoch) + (i + 1)
        lr = get_lr(step_num=step_num)

        batch_input_sequence_indices = model_train_dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_sparse_label = util.label_to_sparse_label(
            model_train_dataset['label'][batch_size * i: batch_size * (i + 1)])

        _, _loss = sess.run([model.minimize, model.loss],
                            {model.input_sequence_indices: batch_input_sequence_indices,
                             model.sparse_label: batch_sparse_label,
                             model.batch_size: min(len(batch_input_sequence_indices), batch_size),
                             model.keep_prob: keep_prob,
                             model.lr: lr,
                             model.tags_loss_weight: tags_loss_weight,
                             model.negative_loss_weight: negative_loss_weight
                             })
        loss += _loss

    return loss / epoch



def validation_ndcg(model, val_util, label_info, batch_size=64):
    dataset = val_util.ndcg_check_dataset
    data_num = len(dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    reco_result = []
    # candidate = []
    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_id_list = dataset['id_list'][batch_size * i: batch_size * (i + 1)]

        reco_songs, reco_tags = sess.run(
            [model.reco_songs, model.reco_tags],
            {model.input_sequence_indices: batch_input_sequence_indices,
             model.keep_prob: 1.0,
             model.song_top_k: 10000,  # 20300,
             model.tag_top_k: 30})  # 1050})

        for _reco_songs, _reco_tags, _id in zip(reco_songs, reco_tags, batch_id_list):
            _reco_songs = label_info.label_encoder.inverse_transform(_reco_songs)
            _reco_tags = label_info.label_encoder.inverse_transform(_reco_tags)

            filtered_songs = []
            for song in _reco_songs:
                if len(filtered_songs) == 100:
                    break

                if song in val_util.plylst_id_seen_songs_dict[_id]:
                    continue
                if song_issue_dict[song] > val_util.plylst_id_plylst_updt_date_dict[_id]:
                    continue
                filtered_songs.append(song)

            if len(filtered_songs) < 100:
                print(len(filtered_songs))

            filtered_tags = list(filter(lambda tag: tag not in val_util.plylst_id_seen_tags_dict[_id], _reco_tags))[:10]
            reco_result.append({'songs': filtered_songs, 'tags': filtered_tags, 'id': _id})

    return val_util._eval(reco_result)


def validation_loss(model, val_util, batch_size=64, tags_loss_weight=0.5, negative_loss_weight=0.5):
    loss = 0

    dataset = val_util.loss_check_dataset
    data_num = len(dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_sparse_label = util.label_to_sparse_label(
            dataset['label'][batch_size * i: batch_size * (i + 1)])

        _loss = sess.run(model.loss,
                         {model.input_sequence_indices: batch_input_sequence_indices,
                          model.sparse_label: batch_sparse_label,
                          model.batch_size: min(len(batch_input_sequence_indices), batch_size),
                          model.keep_prob: 1.0,
                          model.tags_loss_weight: tags_loss_weight,
                          model.negative_loss_weight: negative_loss_weight
                          })
        loss += _loss

    return loss / epoch

def get_songs_tags_embedding_table(model):
    songs_tags_embedding_table = sess.run(model.songs_tags_embedding_table[:model.songs_num+model.tags_num, :])
    return songs_tags_embedding_table

def run(model, sess, train_util, val_util, label_info, saver_path, batch_size=512, keep_prob=0.9,
        tags_loss_weight=0.5, negative_loss_weight=0.5, restore=0):
    if not os.path.exists(saver_path):
        print("create save directory")
        os.makedirs(saver_path)

    if not os.path.exists(os.path.join(saver_path, 'tensorboard')):
        print("create save directory")
        os.makedirs(os.path.join(saver_path, 'tensorboard'))

    if restore != 0:
        model.saver.restore(sess, os.path.join(saver_path, str(restore) + ".ckpt"))
        print('restore:', restore)
    else:
        with tf.name_scope("tensorboard"):
            train_loss_tensorboard = tf.placeholder(tf.float32,
                                                    name='train_loss')  # with regularization (minimize 할 값)
            valid_loss_tensorboard = tf.placeholder(tf.float32, name='valid_loss')  # no regularization
            valid_song_ndcg_tensorboard = tf.placeholder(tf.float32, name='valid_song_ndcg')  # no regularization
            valid_tag_ndcg_tensorboard = tf.placeholder(tf.float32, name='valid_tag_ndcg')  # no regularization
            valid_score_tensorboard = tf.placeholder(tf.float32, name='valid_score')  # no regularization

            train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
            valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
            valid_song_ndcg_summary = tf.summary.scalar("valid_song_ndcg", valid_song_ndcg_tensorboard)
            valid_tag_ndcg_summary = tf.summary.scalar("valid_tag_ndcg", valid_tag_ndcg_tensorboard)
            valid_score_summary = tf.summary.scalar("valid_score", valid_score_tensorboard)

            # merged = tf.summary.merge_all()
            merged_train = tf.summary.merge([train_loss_summary])
            merged_valid = tf.summary.merge(
                [valid_loss_summary, valid_song_ndcg_summary, valid_tag_ndcg_summary, valid_score_summary])

            writer = tf.summary.FileWriter(os.path.join(saver_path, 'tensorboard'), sess.graph)

    epoch_val_score_dict = {}
    for epoch in range(restore + 1, 2):
        train_loss = train(model, train_util, epoch, batch_size=batch_size, keep_prob=keep_prob,
                       tags_loss_weight=tags_loss_weight, negative_loss_weight=negative_loss_weight)
        print("epoch: %d, train_loss: %f" % (epoch, train_loss))
        print()

        # tensorboard
        summary = sess.run(merged_train, {
            train_loss_tensorboard: train_loss})
        writer.add_summary(summary, epoch)

        if (epoch) % 10 == 0 or epoch == 1:
            # tensorboard
            valid_loss = validation_loss(model, val_util, batch_size=batch_size, tags_loss_weight=tags_loss_weight,
                                         negative_loss_weight=negative_loss_weight)
            music_ndcg, tag_ndcg, score = validation_ndcg(model, val_util, label_info, batch_size=batch_size)

            print("epoch: %d, valid_loss: %f, musin_ndcg: %f, tag_ndcg: %f, score: %f" % (
                epoch, valid_loss, music_ndcg, tag_ndcg, score))
            epoch_val_score_dict[epoch] = score

            summary = sess.run(merged_valid, {
                valid_loss_tensorboard: valid_loss,
                valid_song_ndcg_tensorboard: music_ndcg,
                valid_tag_ndcg_tensorboard: tag_ndcg,
                valid_score_tensorboard: score})
            writer.add_summary(summary, epoch)
            print()

            model.saver.save(sess, os.path.join(saver_path, str(epoch) + ".ckpt"))

    epoch_val_score_list = sorted(list(epoch_val_score_dict.items()), key=lambda x:x[1], reverse=True)[:3]
    epoch_val_score_list.append(saver_path)
    util.dump(epoch_val_score_list, os.path.join(parameters.base_dir, 'songs_tags_epoch_val_score_list.pickle'))

    print('restore maximum_score parameters:', epoch_val_score_list[0][1])
    model.saver.restore(sess, os.path.join(saver_path, str(epoch_val_score_list[0][0]) + ".ckpt"))
    songs_tags_embedding_table = get_songs_tags_embedding_table(model)
    util.dump(songs_tags_embedding_table, os.path.join(parameters.base_dir, 'songs_tags_embedding_table.pickle'))


# make train / val set
train_set = util.load_json('dataset/orig/train.json')
val_question = util.load_json('dataset/questions/val.json')
val_answers = util.load_json('dataset/answers/val.json')
song_meta = util.load_json('dataset/song_meta.json')

label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
song_issue_dict = util.load(os.path.join(parameters.base_dir, parameters.song_issue_dict))

train_util = songs_tags_util.TrainSongsTagsUtil(
    train_set, song_meta, parameters.max_sequence_length, label_info)
val_util = songs_tags_util.ValSongsTagsUtil(
    val_question, val_answers, song_meta, parameters.max_sequence_length, label_info)
del train_set, val_question, val_answers, song_meta

# model
model = OrderlessBertAE(
    voca_size=len(label_info.label_encoder.classes_),
    embedding_size=parameters.embed_size,  # 128인경우 850MB 먹음.
    is_embedding_scale=True,
    max_sequence_length=parameters.max_sequence_length,
    encoder_decoder_stack=parameters.stack,
    multihead_num=parameters.multihead,
    pad_idx=label_info.label_encoder.transform([label_info.pad_token])[0],
    songs_num=len(label_info.songs),
    tags_num=len(label_info.tags))

# gpu 할당 및 session 생성
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # nvidia-smi의 k번째 gpu만 사용
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# # 학습 진행
run(
    model,
    sess,
    train_util,
    val_util,
    label_info,
    saver_path='./SONGS_TAGS_emb%d_stack%d_head%d_tags_loss_weight%0.2f_negative_loss_weight%0.2f_bs_%d_warmup_%d' % (
        parameters.embed_size, parameters.stack, parameters.multihead, tags_loss_weight,
        negative_loss_weight, bs, warmup_steps),
    batch_size=bs,
    keep_prob=0.9,
    tags_loss_weight=tags_loss_weight,
    negative_loss_weight=negative_loss_weight,
    restore=0)
