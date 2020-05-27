import os

import implicit

from data_loader.Label_info import Label_info
import data_loader.songs_tags_wmf_util as songs_tags_wmf_util
import parameters
import util



if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')

    label_info_path = os.path.join('./', parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set)

    csr = songs_tags_wmf_util.make_sparse_matrix(  # tags-songs matrix
        train_set,
        label_info.label_encoder,
        len(label_info.songs),
        label_info.label_encoder.transform([label_info.unk_token])[0])
    util.dump(csr, os.path.join('./', parameters.songs_tags_csr_matrix))


    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=parameters.embed_size, use_gpu=False)

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(csr)  # item-user 순으로 넣어야하는데 우리는 tags-songs이므로 item:tags, user:songs
    util.dump(model, os.path.join('./', parameters.songs_tags_wmf))
