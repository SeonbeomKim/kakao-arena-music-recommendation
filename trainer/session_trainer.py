import os

import numpy as np
import tensorflow as tf
from data_loader.Session_interaction import Session
from models.Transformer import Transformer
from tqdm import *
from util import load_json, fill_na


def train(model, session, lr, batch_size=64, keep_prob=0.9, pad_symbol='@pad'):
    loss = 0
    data_num = len(session.model_input_A_length)

    shuffle_index = np.array(range(data_num))
    np.random.shuffle(shuffle_index)

    session.model_input = session.model_input[shuffle_index]
    session.model_input_A_length = session.model_input_A_length[shuffle_index]
    session.positive_label = session.positive_label[shuffle_index]
    session.negative_label = session.negative_label[shuffle_index]

    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm_notebook(range(epoch)):  # TODO tqdm
        input_sequence_indices = session.model_input[batch_size * i: batch_size * (i + 1)]
        # 가장 긴 길이에 맞게 pad

        input_sequence_indices = fill_na(
            input_sequence_indices.tolist(),
            session.label_encoder.transform([pad_symbol])[0]).astype(np.int32)

        try:
            _, train_loss = sess.run([model.minimize, model.loss],
                                     {model.input_sequence_indices: input_sequence_indices,
                                      model.A_length: session.model_input_A_length[
                                                      batch_size * i: batch_size * (i + 1)],
                                      model.next_item_idx: session.positive_label[batch_size * i: batch_size * (i + 1)],
                                      model.negative_item_idx: session.negative_label[
                                                               batch_size * i: batch_size * (i + 1)],
                                      model.keep_prob: keep_prob,
                                      model.lr: lr
                                      })
            loss += train_loss
        except Exception as e:
            print(e)
            print(len(session.model_input[batch_size * i: batch_size * (i + 1)]))
            print(input_sequence_indices.shape)
            print(input_sequence_indices)

    return loss / data_num


def validation(model, session, batch_size=64, pad_symbol='@pad'):
    loss = 0

    data_num = len(session.model_input_A_length)
    epoch = int(np.ceil(data_num / batch_size))

    for i in tqdm(range(epoch)):  # TODO tqdm
        input_sequence_indices = session.model_input[batch_size * i: batch_size * (i + 1)]
        # 가장 긴 길이에 맞게 pad
        input_sequence_indices = fill_na(
            input_sequence_indices.tolist(),
            session.label_encoder.transform([pad_symbol])[0]).astype(np.int32)

        val_loss = sess.run(model.loss,
                            {model.input_sequence_indices: input_sequence_indices,
                             model.A_length: session.model_input_A_length[batch_size * i: batch_size * (i + 1)],
                             model.next_item_idx: session.positive_label[batch_size * i: batch_size * (i + 1)],
                             model.negative_item_idx: session.negative_label[batch_size * i: batch_size * (i + 1)],
                             model.keep_prob: 1.0,
                             })
        loss += val_loss

    return loss / data_num


def run(model, train_session, val_session, saver_path, lr, batch_size=512, keep_prob=0.9, pad_symbol='@pad', restore=0):
    if not os.path.exists(saver_path):
        print("create save directory")
        os.makedirs(saver_path)

    if not os.path.exists(os.path.join(saver_path, 'tensorboard')):
        print("create save directory")
        os.makedirs(os.path.join(saver_path, 'tensorboard'))

    #     if restore != 0:
    #         model.saver.restore(sess, saver_path + str(restore) + ".ckpt")
    #         print('restore:', restore)

    with tf.name_scope("tensorboard"):
        train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')  # with regularization (minimize 할 값)
        valid_loss_tensorboard = tf.placeholder(tf.float32, name='valid_loss')  # no regularization

        train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
        valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(saver_path, 'tensorboard'), sess.graph)
        # merged_train_valid = tf.summary.merge([train_summary, valid_summary])
        # merged_test = tf.summary.merge([test_summary])

    for epoch in range(restore + 1, 150 + 1):
        train_loss = train(model, train_session, lr, batch_size=batch_size, keep_prob=keep_prob, pad_symbol=pad_symbol)
        print("epoch: %d, train_loss: %f" % (epoch, train_loss))
        valid_loss = validation(model, val_session, batch_size=batch_size, pad_symbol=pad_symbol)
        print("epoch: %d, valid_loss: %f" % (epoch, valid_loss))
        print()

        # tensorboard
        summary = sess.run(merged, {
            train_loss_tensorboard: train_loss,
            valid_loss_tensorboard: valid_loss,
        })
        writer.add_summary(summary, epoch)


# make train / val set
train_set = load_json('dataset/orig/train.json')
train_session = Session(train_set)
train_session.set_label_encoder()
train_session.make_dataset(positive_k=3, negative_k=10, sample_num_of_each_plylst=1)

val_set = load_json('dataset/orig/val.json')
val_session = Session(val_set, train_session.label_encoder)
val_session.set_label_encoder()
val_session.make_dataset(positive_k=3, negative_k=10, sample_num_of_each_plylst=1)

# dataset to numpy
train_session.model_input = np.array(train_session.model_input)  # row마다 column수가 달라서 dtype 줄 수 없음
train_session.model_input_A_length = np.array(train_session.model_input_A_length, np.int32)
train_session.positive_label = np.array(train_session.positive_label, np.int32)
train_session.negative_label = np.array(train_session.negative_label, np.int32)

val_session.model_input = np.array(val_session.model_input)
val_session.model_input_A_length = np.array(val_session.model_input_A_length, np.int32)
val_session.positive_label = np.array(val_session.positive_label, np.int32)
val_session.negative_label = np.array(val_session.negative_label, np.int32)

# model
model = Transformer(
    voca_size=len(train_session.label_encoder.classes_),
    embedding_size=12,
    is_embedding_scale=True,
    max_sequence_length=107,
    encoder_decoder_stack=1,
    multihead_num=4,
    pad_idx=train_session.label_encoder.transform(['@pad'])[0],
    cls_idx=train_session.label_encoder.transform(['@cls'])[0])

# gpu 할당 및 session 생성
k = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(k)  # nvidia-smi의 k번째 gpu만 사용
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# # 학습 진행
run(
    model,
    train_session,
    val_session,
    saver_path='./',
    lr=1e-3,
    batch_size=32,
    keep_prob=0.9,
    pad_symbol='@pad',
    restore=0)
