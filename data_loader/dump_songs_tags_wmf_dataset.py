import os

import data_loader.songs_tags_wmf_util as songs_tags_wmf_util
import parameters
import util
from data_loader.Label_info import Label_info

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set)
        util.dump(label_info, label_info_path)

    csr = songs_tags_wmf_util.make_sparse_matrix(  # tags-songs matrix
        train_set,
        label_info.label_encoder,
        len(label_info.songs),
        label_info.label_encoder.transform([label_info.unk_token])[0])
    util.dump(csr, os.path.join(parameters.base_dir, parameters.songs_tags_csr_matrix))
