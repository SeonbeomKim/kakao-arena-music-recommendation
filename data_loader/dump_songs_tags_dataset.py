import os

import data_loader.songs_tags_util as songs_tags_util
import parameters
import util
from data_loader.Label_info import Label_info

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    val_question = util.load_json('dataset/questions/val.json')
    val_answers = util.load_json('dataset/answers/val.json')
    song_meta = util.load_json('dataset/song_meta.json')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set)
        util.dump(label_info, label_info_path)

    val_util = songs_tags_util.ValSongsTagsUtil(val_question, val_answers, song_meta, parameters.max_sequence_length, label_info)
    util.dump(val_util, os.path.join(parameters.base_dir, parameters.songs_tags_transformer_val_sampled_data))
