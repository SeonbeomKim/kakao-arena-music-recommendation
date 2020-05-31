import json

import implicit
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from data_loader.Interaction import Interaction


def export_results(results, playlist_id_encoder, tag_encoder):
    total_result = []
    for idx, result in enumerate(results):
        playlist_dict = {}
        playlist_dict['id'] = playlist_id_encoder.inverse_transform([idx + 1])
        playlist_dict['songs'] = []
        playlist_dict['tags'] = tag_encoder.inverse_transform(result)
        total_result.append(playlist_dict)
    return total_result


def calculate_metric(items, labels):
    dcg = 0.0
    for i, item_id in enumerate(items):
        if item_id in labels:
            dcg += 1.0 / np.log(i + 2)
    ndcg = dcg / (sum((1.0 / np.log(i + 2) for i in range(len(labels)))))
    return ndcg


def process_json_data(val_data, tag_set, song_set):
    for playlist_data in val_data:
        tags = playlist_data['tags']
        playlist_data['tags'] = [tag for tag in tags if tag in tag_set]
        songs = playlist_data['songs']
        playlist_data['songs'] = [song for song in songs if song in song_set]


if __name__ == "__main__":
    lambda_value = 0.1
    neg_sample_ratio = 1.2  # positive dataset 대비 얼마나 neg dataset을 sample 할 지
    learning_rate = 0.05
    batch_size = 256
    epochs = 1

    # loading data
    song_meta_file = "dataset/orig/song_meta.json"
    train_file = "dataset/orig/train.json"
    val_file = "dataset/orig/val.json"
    question_file = "dataset/questions/val.json"
    answer_file = "dataset/answers/val.json"

    with open(train_file, encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, encoding='utf-8') as f:
        val_data = json.load(f)
    with open(question_file, encoding='utf-8') as f:
        question_data = json.load(f)
    with open(answer_file, encoding='utf-8') as f:
        answer_data = json.load(f)

    # build dataset
    train_interaction = Interaction(train_data)
    train_tag_matrix = train_interaction.build_playlist_tag_matrix()
    train_song_matrix = train_interaction.build_playlist_song_matrix()

    train_playlist_encoder = train_interaction.playlist_encoder
    train_tag_encoder = train_interaction.tag_encoder
    train_song_encoder = train_interaction.song_encoder

    process_json_data(val_data, train_interaction.tag_set, train_interaction.song_set)
    val_interaction = Interaction(val_data)
    val_interaction.build_playlist_song_matrix()
    val_interaction.build_playlist_tag_matrix()

    val_tag_playlist_ids = val_interaction.playlist_encoder.transform(val_interaction.raw_tag_playlist_ids)
    val_song_playlist_ids = val_interaction.playlist_encoder.transform(val_interaction.raw_song_playlist_ids)
    val_tags = train_tag_encoder.transform(val_interaction.raw_tags)
    val_songs = train_song_encoder.transform(val_interaction.raw_song)

    tag_data = np.ones((len(val_tags)))
    song_data = np.ones((len(val_songs)))
    val_tag_mat = csr_matrix((tag_data, (val_tag_playlist_ids, val_tags)))
    val_song_mat = csr_matrix((song_data, (val_song_playlist_ids, val_songs)))

    # val_question_interaction = Interaction(question_data)
    # val_question_interaction.build_playlist_tag_matrix()
    # val_question_interaction.build_playlist_song_matrix()

    # question_tag_playlist_ids = train_playlist_encoder.transform(val_question_interaction.raw_tag_playlist_ids)
    # question_song_playlist_ids = train_playlist_encoder.transform(val_question_interaction.raw_song_playlist_ids)
    # question_tags = train_tag_encoder.transform(val_question_interaction.raw_tags)
    # question_songs = train_song_encoder.transform(val_question_interaction.raw_song)

    # tag_data = np.ones((len(question_tags)))
    # song_data = np.ones((len(question_songs)))
    # question_tag_mat = csr_matrix((tag_data, (question_tag_playlist_ids, question_tags)))
    # question_song_mat = csr_matrix((song_data, (question_song_playlist_ids, question_songs)))

    print("building model")
    tag_model = implicit.als.AlternatingLeastSquares(factors=128,
                                                     iterations=10,
                                                     use_gpu=True)
    song_model = implicit.als.AlternatingLeastSquares(factors=128,
                                                      iterations=10,
                                                      use_gpu=True)
    print("training ...")
    tag_model.fit(train_tag_matrix.T.tocsr())
    song_model.fit(train_song_matrix.T.tocsr())

    train_playlist_set = train_interaction.tag_playlist_ids_set
    tag_encoder = train_interaction.tag_encoder
    song_encoder = train_interaction.song_encoder

    submit = []
    # for question, answer in tqdm(zip(question_data, answer_data), desc="predict ...", total=len(question_data)):
    #     question_tags = train_tag_encoder.transform(question['tags'])
    #     question_songs = train_song_encoder.transform(question['songs'])
    #     question_playlist_id = train_playlist_encoder.transform([question['id']])[0]
    #
    #     tag_results = tag_model.recommend(question_playlist_id, question_tag_mat)
    #     tag_predicts = [result[0] for result in tag_results]
    #     tag_predicts = tag_encoder.inverse_transform(tag_predicts)
    #     # tag_ndcg = calculate_metric(tag_predicts, answer['tags'])
    #
    #     song_results = song_model.recommend(question_playlist_id, question_song_mat)
    #     song_predicts = [result[0] for result in song_results]
    #     song_predicts = song_encoder.inverse_transform(song_predicts)
    #     # song_ndcg = calculate_metric(song_predicts, answer['songs'])
    #     # print(tag_ndcg, song_ndcg)
    #     result = {}
    #     result['id'] = question_playlist_id
    #     result['tags'] = tag_predicts
    #     result['songs'] = song_predicts
    #     submit.append(result)

    for data in tqdm(val_data, desc="predict ...", total=len(val_data)):
        question_tags = train_tag_encoder.transform(data['tags'])
        question_songs = train_song_encoder.transform(data['songs'])
        question_playlist_id = val_interaction.playlist_encoder.transform([data['id']])[0]

        tag_results = tag_model.recommend(question_playlist_id, val_tag_mat, recalculate_user=True)
        tag_predicts = [result[0] for result in tag_results]
        tag_predicts = tag_encoder.inverse_transform(tag_predicts)
        # tag_ndcg = calculate_metric(tag_predicts, answer['tags'])

        song_results = song_model.recommend(question_playlist_id, val_song_mat, recalculate_user=True)
        song_predicts = [result[0] for result in song_results]
        song_predicts = song_encoder.inverse_transform(song_predicts)
        # song_ndcg = calculate_metric(song_predicts, answer['songs'])
        # print(tag_ndcg, song_ndcg)
        result = {}
        result['id'] = question_playlist_id
        result['tags'] = tag_predicts
        result['songs'] = song_predicts
        submit.append(result)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(submit, f)
    # results = model.recommend_all(train_matrix, N=10, show_progress=True)
