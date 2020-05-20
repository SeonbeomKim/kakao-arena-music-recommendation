import data_loader.Session_interaction as Session_interaction
import util

if __name__ == "__main__":
    bucket_size = [17, 27, 37, 47, 57, 67, 77, 87, 97, 107]

    train_set = util.load_json('dataset/orig/train.json')
    val_set = util.load_json('dataset/orig/val.json')

    label_encoder = Session_interaction.LabelEncoder(train_set)
    util.dump(label_encoder, './label_encoder.pickle')

    model_train_dataset_bucket = Session_interaction.make_model_dataset_bucket(
        dataset=train_set,
        label_encoder=label_encoder.label_encoder,
        total_songs=label_encoder.total_songs,
        positive_k=3,
        negative_k=10,
        sample_num_of_each_plylst=500,
        bucket_size=bucket_size,
        pad_symbol='@pad')
    util.dump(model_train_dataset_bucket, './model_train_dataset_bucket.pickle')
    del model_train_dataset_bucket

    model_val_dataset_bucket = Session_interaction.make_model_dataset_bucket(
        dataset=val_set,
        label_encoder=label_encoder.label_encoder,
        total_songs=label_encoder.total_songs,
        positive_k=3,
        negative_k=10,
        sample_num_of_each_plylst=500,
        bucket_size=bucket_size,
        pad_symbol='@pad')
    util.dump(model_val_dataset_bucket, './model_val_dataset_bucket.pickle')
