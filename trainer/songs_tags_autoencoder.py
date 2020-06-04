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
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--bs', type=int, default=256)
args.add_argument('--gpu', type=int, default=6)
config = args.parse_args()
lr = config.lr
bs = config.bs
gpu = config.gpu


def train(model, train_input_output, lr, batch_size=64, keep_prob=0.9, tags_loss_weight=0.5, negative_loss_weight=0.5):
    model_train_dataset = songs_tags_util.make_train_val_set(train_input_output, parameters.max_sequence_length,
                                                           label_info, sample=1, shuffle=True)
    loss = 0

    data_num = len(model_train_dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = model_val_dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_A_length = model_val_dataset['A_length'][batch_size * i: batch_size * (i + 1)]
        batch_sparse_label = util.label_to_sparse_label(
            model_val_dataset['label'][batch_size * i: batch_size * (i + 1)])

        _, _loss = sess.run([model.minimize, model.loss],
                            {model.input_sequence_indices: batch_input_sequence_indices,
                             model.A_length: batch_A_length,
                             model.sparse_label: batch_sparse_label,
                             model.batch_size: min(len(batch_input_sequence_indices), batch_size),
                             model.keep_prob: keep_prob,
                             model.lr: lr,
                             model.tags_loss_weight: tags_loss_weight,
                             model.negative_loss_weight: negative_loss_weight
                             })
        loss += _loss

    return loss / data_num



def validation(model, model_val_dataset, batch_size=64, tags_loss_weight=0.5, negative_loss_weight=0.5):
    loss = 0

    data_num = len(model_val_dataset['model_input'])
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):
        batch_input_sequence_indices = model_val_dataset['model_input'][batch_size * i: batch_size * (i + 1)]
        batch_A_length = model_val_dataset['A_length'][batch_size * i: batch_size * (i + 1)]
        batch_sparse_label = util.label_to_sparse_label(
            model_val_dataset['label'][batch_size * i: batch_size * (i + 1)])

        _loss = sess.run(model.loss,
                            {model.input_sequence_indices: batch_input_sequence_indices,
                             model.A_length: batch_A_length,
                             model.sparse_label: batch_sparse_label,
                             model.batch_size:min(len(batch_input_sequence_indices), batch_size),
                             model.keep_prob: 1.0,
                             model.tags_loss_weight: tags_loss_weight,
                             model.negative_loss_weight: negative_loss_weight
                             })
        loss += _loss

    return loss / data_num


def run(model, sess, train_input_output, model_val_dataset, saver_path, lr, batch_size=512, keep_prob=0.9,
        tags_loss_weight=0.5, negative_loss_weight=0.5, restore=0):
    if not os.path.exists(saver_path):
        print("create save directory")
        os.makedirs(saver_path)

    if not os.path.exists(os.path.join(saver_path, 'tensorboard')):
        print("create save directory")
        os.makedirs(os.path.join(saver_path, 'tensorboard'))

    if restore != 0:
        model.saver.restore(sess, saver_path + str(restore) + ".ckpt")
        print('restore:', restore)
    else:
        with tf.name_scope("tensorboard"):
            train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')  # with regularization (minimize 할 값)
            valid_loss_tensorboard = tf.placeholder(tf.float32, name='valid_loss')  # no regularization
            # valid_positive_loss_tensorboard = tf.placeholder(tf.float32, name='valid_positive_loss')  # no regularization

            train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
            valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
            # valid_positive_loss_summary = tf.summary.scalar("valid_positive_loss", valid_positive_loss_tensorboard)

            # merged = tf.summary.merge_all()
            merged_train = tf.summary.merge([train_loss_summary])
            merged_valid = tf.summary.merge([valid_loss_summary])
            writer = tf.summary.FileWriter(os.path.join(saver_path, 'tensorboard'), sess.graph)

    for epoch in range(restore + 1, 151):
        train_loss = train(model, train_input_output, lr, batch_size=batch_size, keep_prob=keep_prob,
                           tags_loss_weight=tags_loss_weight, negative_loss_weight=negative_loss_weight)
        print("epoch: %d, train_loss: %f" % (epoch, train_loss))
        print()

        # tensorboard
        summary = sess.run(merged_train, {
            train_loss_tensorboard: train_loss})
        writer.add_summary(summary, epoch)

        if (epoch) % 5 == 0:
            # tensorboard
            valid_loss = validation(model, model_val_dataset, batch_size=batch_size, tags_loss_weight=tags_loss_weight,
                                    negative_loss_weight=negative_loss_weight)
            print("epoch: %d, valid_loss: %f" % (epoch, valid_loss))

            summary = sess.run(merged_valid, {
                valid_loss_tensorboard: valid_loss})
            writer.add_summary(summary, epoch)
            print()

            model.saver.save(sess, os.path.join(saver_path, str(epoch) + ".ckpt"))


def make_transformer_embedding(songs_embedding, tags_embedding, label_info):
    pad_token = label_info.pad_token
    unk_token = label_info.unk_token
    others_embedding_tokens = label_info.others_for_encoder + [unk_token]

    embed_size = songs_embedding.shape[1]

    others_embedding = np.random.randn(len(others_embedding_tokens), embed_size)
    others_embedding[others_embedding_tokens.index(pad_token)] *= 0

    transformer_embedding = np.concatenate((songs_embedding, tags_embedding, others_embedding)).astype(np.float32)
    return transformer_embedding


# make train / val set
train_input_output = util.load(os.path.join(parameters.base_dir, parameters.songs_tags_transformer_train_input_output))
model_val_dataset = util.load(os.path.join(parameters.base_dir, parameters.songs_tags_transformer_val_sampled_data))
label_info = util.load(os.path.join(parameters.base_dir, parameters.label_info))

songs_tags_wmf = util.load(os.path.join(parameters.base_dir, parameters.songs_tags_wmf))
init_embedding = make_transformer_embedding(songs_tags_wmf.user_factors, songs_tags_wmf.item_factors, label_info)

# model
model = OrderlessBertAE(
    voca_size=len(label_info.label_encoder.classes_),
    embedding_size=parameters.embed_size,  # 128인경우 850MB 먹음.
    is_embedding_scale=True,
    max_sequence_length=parameters.max_sequence_length,
    encoder_decoder_stack=parameters.stack,
    multihead_num=parameters.multihead,
    pad_idx=label_info.label_encoder.transform([label_info.pad_token])[0],
    unk_idx=label_info.label_encoder.transform([label_info.unk_token])[0],
    songs_num=len(label_info.songs),
    tags_num=len(label_info.tags))

# gpu 할당 및 session 생성
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # nvidia-smi의 k번째 gpu만 사용
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(model.song_tag_embedding_table.assign(init_embedding))  # wmf로 pretraining된 임베딩 사용

tags_loss_weight = 0.55
negative_loss_weight = 0.55
# # 학습 진행
run(
    model,
    sess,
    train_input_output,
    model_val_dataset,
    saver_path='./saver_decay_new_AE_batch_emb%d_stack%d_head%d_lr_%0.5f_tags_loss_weight_%0.2f_negative_loss_weight_%0.2f' % (
        parameters.embed_size, parameters.stack, parameters.multihead, lr, tags_loss_weight, negative_loss_weight),
    lr=lr,
    batch_size=bs,
    keep_prob=0.9,
    tags_loss_weight=tags_loss_weight,
    negative_loss_weight=negative_loss_weight,
    restore=0)
