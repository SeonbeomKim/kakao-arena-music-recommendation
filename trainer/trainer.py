import json

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader.Interaction import Interaction, sample_neg
from models.WMF import WMF

tf.random.set_seed(777)


def export_results(results, playlist_id_encoder, tag_encoder):
    total_result = []
    for idx, result in enumerate(results):
        playlist_dict = {}
        playlist_dict['id'] = playlist_id_encoder.inverse_transform([idx+1])
        playlist_dict['songs'] = []
        playlist_dict['tags'] = tag_encoder.inverse_transform(result)
        total_result.append(playlist_dict)
    return total_result


if __name__ == "__main__":
    lambda_value = 0.1
    neg_sample_ratio = 1.2  # positive dataset 대비 얼마나 neg dataset을 sample 할 지
    learning_rate = 0.05
    batch_size = 256
    epochs = 1

    # loading data
    print("loading data ...")
    song_meta_file = "../res/song_meta.json"
    train_file = "../res/train.json"
    with open(train_file) as f:
        data = json.load(f)

    # build dataset
    print("building dataset ...")
    interaction = Interaction(data)
    matrix = interaction.build_interaction_matrix()
    neg_matrix = sample_neg(matrix, neg_sample_ratio=1.2)

    U_size = interaction.num_playlist_ids
    V_size = interaction.num_tags
    train_dataset, test_dataset = train_test_split(matrix, test_size=0.2, random_state=42)
    train_neg_dataset, test_neg_dataset = train_test_split(neg_matrix, test_size=0.2, random_state=42)

    print("building model")
    wmf = WMF(
        U_size,
        V_size,
        embedding_size=150,
        regularization_factor=1.5,
        alpha=40.,
        learning_rate=0.0001
    )
    wmf.build_network()
    print("training ...")
    wmf.train(
        train_dataset,
        train_neg_dataset,
        batch_size=batch_size,
        epochs=epochs)
    results = wmf.test(test_dataset, test_neg_dataset)
    export_results(results, interaction.playlist_encoder, interaction.tag_encoder)
