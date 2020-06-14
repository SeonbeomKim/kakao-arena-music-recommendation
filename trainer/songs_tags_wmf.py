import os

import implicit

from data_loader.Label_info import Label_info
import parameters
import util



if __name__ == "__main__":
    label_info_path = os.path.join('./', parameters.label_info)
    label_info = util.load(label_info_path)
    csr = util.load(os.path.join('./', parameters.songs_tags_csr_matrix))

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=parameters.embed_size, use_gpu=False)

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(csr)  # item-user 순으로 넣어야하는데 우리는 tags-songs이므로 item:tags, user:songs
    util.dump(model, os.path.join('./', parameters.songs_tags_wmf))
