import os

import numpy as np
import tensorflow as tf
import util
# from data_loader.Session_interaction import Session
from models.Transformer import Transformer
from tqdm import tqdm

import parameters

def train(model, session, lr, batch_size=64, keep_prob=0.9):
    loss = 0

    total_data_num = 0
    for bucket in session:
        data_num = len(session[bucket]['model_input_A_length'])
        total_data_num += data_num
        shuffle_index = np.array(range(data_num))
        np.random.shuffle(shuffle_index)

        session[bucket]['model_input'] = session[bucket]['model_input'][shuffle_index]
        session[bucket]['model_input_A_length'] = session[bucket]['model_input_A_length'][shuffle_index]
        session[bucket]['positive_label'] = session[bucket]['positive_label'][shuffle_index]
        session[bucket]['negative_label'] = session[bucket]['negative_label'][shuffle_index]

        epoch = int(np.ceil(data_num / batch_size))
        for i in tqdm(range(epoch)):
            _, train_loss = sess.run([model.minimize, model.loss],
                                     {model.input_sequence_indices: session[bucket]['model_input'][
                                                                    batch_size * i: batch_size * (i + 1)],
                                      model.A_length: session[bucket]['model_input_A_length'][
                                                      batch_size * i: batch_size * (i + 1)],
                                      model.next_item_idx: session[bucket]['positive_label'][
                                                           batch_size * i: batch_size * (i + 1)],
                                      model.negative_item_idx: session[bucket]['negative_label'][
                                                               batch_size * i: batch_size * (i + 1)],
                                      model.keep_prob: keep_prob,
                                      model.lr: lr
                                      })
            loss += train_loss

    return loss / total_data_num


def validation(model, session, batch_size=64):
    loss = 0
    positive_loss = 0
    total_data_num = 0
    for bucket in session:
        data_num = len(session[bucket]['model_input_A_length'])
        total_data_num += data_num

        epoch = int(np.ceil(data_num / batch_size))
        for i in tqdm(range(epoch)):
            val_loss, val_positive_loss = sess.run([model.loss, model.positive_loss],
                                {model.input_sequence_indices: session[bucket]['model_input'][
                                                               batch_size * i: batch_size * (i + 1)],
                                 model.A_length: session[bucket]['model_input_A_length'][
                                                 batch_size * i: batch_size * (i + 1)],
                                 model.next_item_idx: session[bucket]['positive_label'][
                                                      batch_size * i: batch_size * (i + 1)],
                                 model.negative_item_idx: session[bucket]['negative_label'][
                                                          batch_size * i: batch_size * (i + 1)],
                                 model.keep_prob: 1.0,
                                 })
            loss += val_loss
            positive_loss += val_positive_loss

    return loss / total_data_num, positive_loss / total_data_num


def run(model, sess, saver, train_session, val_session, saver_path, lr, batch_size=512, keep_prob=0.9, restore=0):
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
            valid_positive_loss_tensorboard = tf.placeholder(tf.float32, name='valid_positive_loss')  # no regularization

            train_loss_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
            valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_tensorboard)
            valid_positive_loss_summary = tf.summary.scalar("valid_positive_loss", valid_positive_loss_tensorboard)

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(os.path.join(saver_path, 'tensorboard'), sess.graph)
            # merged_train_valid = tf.summary.merge([train_summary, valid_summary])
            # merged_test = tf.summary.merge([test_summary])

    for epoch in range(restore + 1, 20):
        train_loss = train(model, train_session, lr, batch_size=batch_size, keep_prob=keep_prob)
        print("epoch: %d, train_loss: %f" % (epoch, train_loss))
        valid_loss, valid_positive_loss = validation(model, val_session, batch_size=batch_size)
        print("epoch: %d, valid_loss: %f, valid_positive_loss: %f" % (epoch, valid_loss, valid_positive_loss))
        print()

        # tensorboard
        summary = sess.run(merged, {
            train_loss_tensorboard: train_loss,
            valid_loss_tensorboard: valid_loss,
            valid_positive_loss_tensorboard: valid_positive_loss
        })
        writer.add_summary(summary, epoch)
        # if (epoch) % 5 == 0:
        print('save model')
        saver.save(sess, os.path.join(saver_path, str(epoch) + ".ckpt"))


def make_transformer_embedding(songs_embedding, tags_embedding, others_for_encoder):
    embed_size = songs_embedding.shape[1]

    others_embedding = np.random.randn(len(others_for_encoder + ['@unk']), embed_size)
    others_embedding[others_for_encoder.index('@pad')] *= 0

    transformer_embedding = np.concatenate((songs_embedding, tags_embedding, others_embedding)).astype(np.float32)
    return transformer_embedding


# make train / val set
train_session = util.load('./model_train_dataset_bucket_recall.pickle')
val_session = util.load('./model_val_dataset_bucket_recall.pickle')
label_encoder = util.load('./label_encoder.pickle')
total_songs = label_encoder.songs  # 나중에 playlist 임베딩이랑 내적할때 total_songs에 대해서만 수행
others_for_encoder = label_encoder.others_for_encoder
label_encoder = label_encoder.label_encoder
pad_symbol = '@pad'

tags_songs_wmf = util.load('./tags_songs_wmf.pickle')
init_embedding = make_transformer_embedding(tags_songs_wmf.user_factors, tags_songs_wmf.item_factors,
                                            others_for_encoder)

# model
model = Transformer(
    voca_size=len(label_encoder.classes_),
    embedding_size=parameters.embed_size,  # 128인경우 850MB 먹음.
    is_embedding_scale=True,
    max_sequence_length=parameters.max_sequence_length,
    encoder_decoder_stack=parameters.stack,
    multihead_num=parameters.multihead,
    pad_idx=label_encoder.transform(['@pad'])[0],
    unk_idx=label_encoder.transform(['@unk'])[0])

# gpu 할당 및 session 생성
k = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(k)  # nvidia-smi의 k번째 gpu만 사용
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 gpu 메모리 사용

sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=10000)

sess.run(tf.global_variables_initializer())
sess.run(model.embedding_table.assign(init_embedding)) # wmf로 pretraining된 임베딩 사용

# # 학습 진행
run(
    model,
    sess,
    saver,
    train_session,
    val_session,
    saver_path='./saver_emb%d_stack%d_head%d_lr_2e-5_recall_pre_MaskUnkLabel' % (parameters.embed_size, parameters.stack, parameters.multihead),
    lr=2e-5,
    batch_size=512,
    keep_prob=0.9,
    restore=0)
