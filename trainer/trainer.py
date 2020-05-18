import json
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.WMF import WMF
from data_loader.Interaction import Interaction, sample_neg

tf.random.set_seed(777)

if __name__ == "__main__":
    lambda_value = 0.1
    neg_sample_ratio = 1.2  # positive dataset 대비 얼마나 neg dataset을 sample 할 지
    learning_rate = 0.05
    batch_size = 256
    epochs = 100

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
    wmf.test(test_dataset, test_neg_dataset)
