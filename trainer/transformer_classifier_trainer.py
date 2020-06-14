# coding=utf-8

import os
from collections import Counter

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tqdm.auto import tqdm

from models.TransformerClassifier import Classifier, TagClassifierDataGenerator, loss_function
from util import load_json

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)

tf.config.experimental_run_functions_eagerly(True)

TRAIN_DATA_PATH = '../res/orig/train.json'
META_DATA_PATH = '../res/song_meta.json'
VAL_Q_DATA_PATH = '../res/questions/val.json'
VAL_A_DATA_PATH = '../res/answers/val.json'
TEST_DATA_PATH = '../res/test.json'

MAX_SONG_CNT = 200000
MAX_TAG_CNT = 20000
MAX_SEQ_LEN = 96
MAX_OUT_LEN = 4
EMBED_DIM = 128  # Embedding size for each token
NUM_HEADS = 4  # Number of attention heads
FF_DIM = 128  # Hidden layer size in feed forward network inside transformer
FC_SIZE = 256
BATCH_SIZE = 128
MAX_EPOCHES = 50


def playlist_to_songs(playlists):
    song_seqs = []
    for plays in tqdm(playlists):
        song_seqs.append(list(map(lambda s: str(s), plays['songs'])))
    return song_seqs


def playlist_to_tags(playlists):
    song_seqs = []
    for plays in tqdm(playlists):
        song_seqs.append(plays['tags'])
    return song_seqs


def filter_train_data(data, candidate_set):
    filtered_data = []
    for train in data:
        valid_tags = filter(lambda t: t in candidate_set, train['tags'])
        train['tags'] = list(valid_tags)
        if len(train['tags']) > 0:
            filtered_data.append(train)

    return filtered_data


if __name__ == '__main__':
    # 트레이닝 데이터 로드
    train_data = load_json(TRAIN_DATA_PATH)

    # tag 필터링
    tags_in_train = [tag for tags in map(lambda d: filter(lambda t: t, d['tags']), train_data) for tag in tags]
    tag_counter = Counter(tags_in_train).most_common()
    candidate_tags = set(map(lambda tc: tc[0], tag_counter[:MAX_TAG_CNT]))

    # tag 기준으로 트레이닝 데이터셋 필터링
    filtered_train = filter_train_data(data=train_data, candidate_set=candidate_tags)

    # 필터링된 데이터셋에 있는 노래 리스트
    song_lists = playlist_to_songs(filtered_train)
    # 노래 id tokenize
    song_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=MAX_SONG_CNT)
    song_tokenizer.fit_on_texts(song_lists)

    # 필터링된 데이터셋에 있는 태그
    tag_lists = playlist_to_tags(filtered_train)
    # 태그 labeling
    tag_encoder = MultiLabelBinarizer()
    tag_encoder.fit(map(lambda d: d['tags'], filtered_train))

    train_song, val_song, train_tag, val_tag = train_test_split(song_lists, tag_lists, test_size=0.2)

    # Data generator 생성
    train_generator = TagClassifierDataGenerator(
        song_list=train_song,
        tag_list=train_tag,
        song_encoder=song_tokenizer,
        tag_encoder=tag_encoder,
        max_song_seq=MAX_SEQ_LEN,
        max_tag_size=MAX_TAG_CNT,
        batch_size=BATCH_SIZE
    )
    val_generator = TagClassifierDataGenerator(
        song_list=val_song,
        tag_list=val_tag,
        song_encoder=song_tokenizer,
        tag_encoder=tag_encoder,
        max_song_seq=MAX_SEQ_LEN,
        max_tag_size=MAX_TAG_CNT,
        batch_size=BATCH_SIZE
    )

    # 모델 생성
    transformer = Classifier(num_layers=NUM_HEADS, num_heads=NUM_HEADS, dff=FF_DIM,
                             d_model=EMBED_DIM, input_vocab_size=MAX_SONG_CNT,
                             target_vocab_size=MAX_TAG_CNT)

    # lr, optimizer 생성
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    # 모델 빌드
    clf_model = transformer.build_model(max_seq=MAX_SEQ_LEN, fc_size=FC_SIZE, training=True)
    clf_model.compile(loss=loss_function, optimizer=optimizer,
                      metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    # 트레이닝 수행
    clf_model.fit(train_generator, epochs=MAX_EPOCHES)

    # TODO: 모델 성능 테스트
