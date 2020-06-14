import os

import data_loader.re_rank_util as re_rank_util
import parameters
import util
from data_loader.Label_info import Label_info

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    val_question = util.load_json('dataset/questions/val.json')
    val_answers = util.load_json('dataset/answers/val.json')
    song_meta = util.load_json('dataset/song_meta.json')
    reco_result = util.load('saver_MAKE_RECO_FILE_no_pre_v3_song_tag_decay0.001_emb128_stack4_head4_lr_0.00070_tags_loss_weight_0.30_negative_loss_weight_0.05/40reco_result.pickle')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set, song_meta)
        util.dump(label_info, label_info_path)


    val_util = re_rank_util.ValReRankUtil(val_question, val_answers, song_meta,
                                          reco_result, parameters.max_sequence_length, label_info)
    util.dump(val_util, os.path.join(parameters.base_dir, 're_rank.val.pickle'))
