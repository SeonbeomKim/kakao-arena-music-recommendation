import os
from glob import glob

import argparse
import numpy as np
import parameters
import sentencepiece as spm
import tensorflow as tf
from data_loader.plylst_title_util import TrainUtil, ValUtil
from evaluate import ArenaEvaluator
from models.TitleBert import TitleBert
from tqdm import tqdm

import util

args = argparse.ArgumentParser()
args.add_argument('--bs', type=int, default=128)
args.add_argument('--gpu', type=int, default=6)
args.add_argument('--tags_loss_weight', type=float, default=0.15)
args.add_argument('--warmup_steps', type=float, default=4000)

config = args.parse_args()
bs = config.bs
gpu = config.gpu
tags_loss_weight = config.tags_loss_weight
warmup_steps = config.warmup_steps


def get_lr(step_num):
    '''
    https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf
    step_num(training_steps):  number of iterations, ie. the number of times the optimizer update was run
        This number also equals the number of mini batches that were processed.
    '''
    lr = (parameters.title_model_embed_size ** -0.5) * min((step_num ** -0.5), (step_num * (warmup_steps ** -1.5)))
    return lr


def train(model, train_util, iter, batch_size=64, keep_prob=0.9, tags_loss_weight=0.15):
    if iter < 50:
        minimize_func = model.minimize
    else:
        minimize_func = model.minimize_with_ranking_loss

    model_train_dataset = train_util.make_dataset(shuffle=True)
    loss = 0

    data_num = len(model_train_dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        step_num = ((iter - 1) * epoch) + (i + 1)
        lr = get_lr(step_num=step_num)

        batch_input_sequence_indices = model_train_dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_sparse_label = util.label_to_sparse_label(
            model_train_dataset['label'][batch_size * i: batch_size * (i + 1)])

        _, _loss = sess.run([minimize_func, model.loss],
                            {model.input_sequence_indices: batch_input_sequence_indices,
                             model.sparse_label: batch_sparse_label,
                             model.batch_size: min(len(batch_input_sequence_indices), batch_size),
                             model.keep_prob: keep_prob,
                             model.lr: lr,
                             model.tags_loss_weight: tags_loss_weight})
        loss += _loss

    return loss / epoch


def pre_train_masked_LM(model, train_util, iter, batch_size=64, keep_prob=0.9):
    pre_train_dataset = train_util.make_pre_train_dataset(shuffle=True)
    loss = 0

    data_num = len(pre_train_dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        step_num = ((iter - 1) * epoch) + (i + 1)
        lr = get_lr(step_num=step_num)

        batch_input_sequence_indices = pre_train_dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_mask_label = []
        for each in pre_train_dataset['mask_label'][batch_size * i: batch_size * (i + 1)]:
            batch_mask_label += each
        batch_boolean_mask = pre_train_dataset['boolean_mask'][batch_size * i: batch_size * (i + 1)]

        _, _loss = sess.run([model.minimize_masked_LM_loss, model.masked_LM_loss],
                            {model.input_sequence_indices: batch_input_sequence_indices,
                             model.boolean_mask: batch_boolean_mask,
                             model.masked_LM_label: batch_mask_label,
                             model.keep_prob: keep_prob,
                             model.lr: lr})
        loss += _loss

    return loss / epoch


def validation_pre_train_masked_LM(model, val_util, batch_size=64):
    loss = 0
    accuracy = 0
    total_masked_LM_data = 0

    dataset = val_util.pre_train_accuracy_check_dataset
    data_num = len(dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_mask_label = []
        for each in dataset['mask_label'][batch_size * i: batch_size * (i + 1)]:
            batch_mask_label += each
        batch_boolean_mask = dataset['boolean_mask'][batch_size * i: batch_size * (i + 1)]

        total_masked_LM_data += len(batch_mask_label)
        _loss, _accuracy = sess.run([model.masked_LM_loss, model.masked_LM_correct],
                                    {model.input_sequence_indices: batch_input_sequence_indices,
                                     model.boolean_mask: batch_boolean_mask,
                                     model.masked_LM_label: batch_mask_label,
                                     model.keep_prob: 1.0
                                     })
        loss += _loss
        accuracy += _accuracy
    return loss / epoch, accuracy / total_masked_LM_data


def validation_ndcg(model, val_util, label_info, batch_size=64):
    dataset = val_util.ndcg_check_dataset
    data_num = len(dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    reco_result = []
    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_id_list = dataset['id_list'][batch_size * i: batch_size * (i + 1)]
        batch_seen_songs_set = dataset['seen_songs_set'][batch_size * i: batch_size * (i + 1)]
        batch_seen_tags_set = dataset['seen_tags_set'][batch_size * i: batch_size * (i + 1)]
        batch_plylst_updt_date = dataset['plylst_updt_date'][batch_size * i: batch_size * (i + 1)]

        reco_songs, reco_tags = sess.run(
            [model.reco_songs, model.reco_tags],
            {model.input_sequence_indices: batch_input_sequence_indices,
             model.keep_prob: 1.0,
             model.song_top_k: 10000,  # 20300,
             model.tag_top_k: 30})  # 1050})

        for k in range(len(reco_songs)):
            _reco_songs = label_info.label_encoder.inverse_transform(reco_songs[k])
            _reco_tags = label_info.label_encoder.inverse_transform(reco_tags[k])
            _id = batch_id_list[k]

            filtered_songs = []
            for song in _reco_songs:
                if len(filtered_songs) == 100:
                    break

                if song in batch_seen_songs_set[k]:
                    continue
                if song_issue_dict[song] > batch_plylst_updt_date[k]:
                    continue
                filtered_songs.append(song)

            if len(filtered_songs) < 100:
                print(len(filtered_songs))

            filtered_tags = list(filter(lambda tag: tag not in batch_seen_tags_set[k], _reco_tags))[:10]
            reco_result.append({'songs': filtered_songs, 'tags': filtered_tags, 'id': _id})

    return evaluator.evaluate_from_data(dataset['gt'], reco_result)


def validation_loss(model, val_util, batch_size=64, tags_loss_weight=0.15):
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
                          model.tags_loss_weight: tags_loss_weight})
        loss += _loss

    return loss / epoch


def save_model(model, sess, path, epoch):
    for each in glob(os.path.join(path, '*')):
        if 'tensorboard' in each:
            continue
        if 'checkpoint' in each:
            continue
        print('rm %s' % each)
        os.remove(each)

    new_best_model_path = os.path.join(path, '%d.ckpt' % epoch)
    print('save new best_model: %s' % new_best_model_path)
    model.saver.save(sess, new_best_model_path)


def run(model, sess, train_util, val_util, label_info, saver_path, batch_size=512, keep_prob=0.9,
        tags_loss_weight=0.15):
    if not os.path.exists(saver_path):
        print("create save directory")
        os.makedirs(saver_path)

    if not os.path.exists(os.path.join(saver_path, 'tensorboard')):
        print("create save directory")
        os.makedirs(os.path.join(saver_path, 'tensorboard'))

    with tf.name_scope("tensorboard"):
        train_loss_tensorboard = tf.placeholder(tf.float32,
                                                name='train_loss')  # with regularization (minimize ??媛?
        valid_loss_tensorboard = tf.placeholder(tf.float32, name='valid_loss')  # no regularization
        valid_song_ndcg_tensorboard = tf.placeholder(tf.float32, name='valid_song_ndcg')  # no regularization
        valid_tag_ndcg_tensorboard = tf.placeholder(tf.float32, name='valid_tag_ndcg')  # no regularization
        valid_score_tensorboard = tf.placeholder(tf.float32, name='valid_score')  # no regularization

        pre_train_loss_tensorboard = tf.placeholder(tf.float32,
                                                    name='pre_train_loss')  # with regularization (minimize ??媛?
        pre_valid_loss_tensorboard = tf.placeholder(tf.float32, name='pre_valid_loss')  # no regularization

        pre_train_loss_summary = tf.summary.scalar("pre_train_loss", pre_train_loss_tensorboard)
        pre_valid_loss_summary = tf.summary.scalar("pre_valid_loss", pre_valid_loss_tensorboard)

        train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
        valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
        valid_song_ndcg_summary = tf.summary.scalar("valid_song_ndcg", valid_song_ndcg_tensorboard)
        valid_tag_ndcg_summary = tf.summary.scalar("valid_tag_ndcg", valid_tag_ndcg_tensorboard)
        valid_score_summary = tf.summary.scalar("valid_score", valid_score_tensorboard)

        # merged = tf.summary.merge_all()
        merged_train = tf.summary.merge([train_loss_summary])
        merged_valid = tf.summary.merge(
            [valid_loss_summary, valid_song_ndcg_summary, valid_tag_ndcg_summary, valid_score_summary])

        merged_pre_train = tf.summary.merge([pre_train_loss_summary])
        merged_pre_valid = tf.summary.merge([pre_valid_loss_summary])

        writer = tf.summary.FileWriter(os.path.join(saver_path, 'tensorboard'), sess.graph)

    # pretrain
    for epoch in range(1, 101):
        pre_train_loss = pre_train_masked_LM(model, train_util, epoch, batch_size=batch_size, keep_prob=keep_prob)
        print('pre_train_loss_masked_LM epoch: %d, pre_train_loss: %f' % (epoch, pre_train_loss))

        valid_loss, accuracy = validation_pre_train_masked_LM(model, val_util, batch_size=batch_size)
        print('pre_train_valid_loss: %f, pre_train_valid_accuracy: %f' % (valid_loss, accuracy))
        print()

    best_model_dict = {'epoch': 0, 'score': 0, 'saver_path': saver_path}
    for epoch in range(1, 301):
        train_loss = train(model, train_util, epoch, batch_size=batch_size, keep_prob=keep_prob,
                           tags_loss_weight=tags_loss_weight)
        print("TITLE epoch: %d, train_loss: %f" % (epoch, train_loss))
        print()

        # tensorboard
        summary = sess.run(merged_train, {
            train_loss_tensorboard: train_loss})
        writer.add_summary(summary, epoch)

        if (epoch) % 5 == 0 or epoch == 1:
            # tensorboard
            valid_loss = validation_loss(model, val_util, batch_size=batch_size, tags_loss_weight=tags_loss_weight)
            music_ndcg, tag_ndcg, score = validation_ndcg(model, val_util, label_info, batch_size=batch_size)
            print("epoch: %d, valid_loss: %f, musin_ndcg: %f, tag_ndcg: %f, score: %f" % (
                epoch, valid_loss, music_ndcg, tag_ndcg, score))

            summary = sess.run(merged_valid, {
                valid_loss_tensorboard: valid_loss,
                valid_song_ndcg_tensorboard: music_ndcg,
                valid_tag_ndcg_tensorboard: tag_ndcg,
                valid_score_tensorboard: score})
            writer.add_summary(summary, epoch)
            print()

            if score > best_model_dict['score']:
                best_model_dict['score'] = score
                best_model_dict['epoch'] = epoch
                save_model(model, sess, saver_path, epoch)
                util.dump(best_model_dict, os.path.join(parameters.base_dir, 'plylst_title_best_model_dict.pickle'))

            print('best_model_epoch: %d, best_model_score: %f' % (best_model_dict['epoch'], best_model_dict['score']))

            # 50번동안 최고 성적 안나왔으면 멈춤
            if epoch >= best_model_dict['epoch'] + 50:
                print('early stopping')
                break


# make train / val set
train_set = util.load_json('dataset/orig/train.json')
val_question = util.load_json('dataset/questions/val.json')
val_answers = util.load_json('dataset/answers/val.json')

label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))
song_issue_dict = util.load(os.path.join(parameters.base_dir, parameters.song_issue_dict))

sp = spm.SentencePieceProcessor(model_file=os.path.join(parameters.base_dir, parameters.bpe_model_file))

train_util = TrainUtil(train_set, parameters.title_model_max_sequence_length, label_info, sp)
val_util = ValUtil(val_question, val_answers, parameters.title_model_max_sequence_length, label_info, sp)
del train_set, val_question, val_answers

evaluator = ArenaEvaluator()

# model
print('voca_size: %d' % len(sp))

model = TitleBert(
    voca_size=len(sp),
    embedding_size=parameters.title_model_embed_size,
    is_embedding_scale=True,
    max_sequence_length=parameters.title_model_max_sequence_length,
    encoder_decoder_stack=parameters.title_model_stack,
    multihead_num=parameters.title_model_multihead,
    pad_idx=sp.piece_to_id(parameters.pad_token),
    songs_num=len(label_info.songs),
    tags_num=len(label_info.tags))

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

run(
    model,
    sess,
    train_util,
    val_util,
    label_info,
    saver_path='./TITLE_VOCA_%d_emb%d_stack%d_head%d_tags_loss_weight%0.2f_bs_%d_warmup_%d' % (
        parameters.bpe_voca_size, parameters.title_model_embed_size, parameters.title_model_stack,
        parameters.title_model_multihead, tags_loss_weight, bs, warmup_steps),
    batch_size=bs,
    keep_prob=0.9,
    tags_loss_weight=tags_loss_weight)
