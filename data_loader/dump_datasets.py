import os

import data_loader.plylst_title_util as plylst_title_util
import parameters
import util
from data_loader.Label_info import Label_info
from split_data import ArenaSplitter

# dataset 분리
splitter = ArenaSplitter()
splitter.run('dataset/train.json')


train_set = util.load_json('dataset/orig/train.json')
song_meta = util.load_json('dataset/song_meta.json')

# make label_info
label_info = Label_info(
    train_set, song_meta, songs_recall=parameters.songs_recall, tags_recall=parameters.tags_recall)
util.dump(label_info, os.path.join(parameters.base_dir, parameters.label_info))

# make sentencepiece (os.path.join(parameters.base_dir, parameters.bpe_model_file)) 에 저장됨
plylst_title_util.train_sentencepiece(train_set)


_train_set = util.load_json('dataset/train.json')
song_issue_dict = util.get_song_issue_dict(_train_set, song_meta, label_info)
util.dump(song_issue_dict, os.path.join(parameters.base_dir, parameters.song_issue_dict))

