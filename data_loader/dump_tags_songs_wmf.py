import data_loader.Session_interaction as Session_interaction
import implicit
import util
from scipy.sparse import csr_matrix

import parameters

def make_sparse_matrix(dataset, label_encoder, songs_num, tags_num, unk_idx):
    tags_songs_dict = {}
    for each in dataset:
        songs = each['songs']
        tags = each['tags']

        for tag in tags:
            tag_idx = label_encoder.transform([tag])[0]
            if tag_idx == unk_idx:
                continue
            valid_tag_idx = tag_idx - songs_num  # song_num개수가 tags의 시작 idx 이므로 songs_num을 빼줘야 idx 0부터 시작함.

            for song in songs:
                song_idx = label_encoder.transform([song])[0]
                if song_idx == unk_idx:
                    continue

                if (valid_tag_idx, song_idx) not in tags_songs_dict:
                    tags_songs_dict[(valid_tag_idx, song_idx)] = 0
                tags_songs_dict[(valid_tag_idx, song_idx)] += 1

    tags = []
    songs = []
    co_occur = []
    for tags_songs in tags_songs_dict:
        tag, song = tags_songs
        co = tags_songs_dict[tags_songs]

        tags.append(tag)
        songs.append(song)
        co_occur.append(co)

    csr = csr_matrix((co_occur, (tags, songs)))
    return csr


if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')

    label_encoder = Session_interaction.LabelEncoder(train_set)
    util.dump(label_encoder, './label_encoder.pickle')

    csr = make_sparse_matrix(  # tags-songs matrix
        train_set,
        label_encoder.label_encoder,
        len(label_encoder.songs),
        len(label_encoder.tags),
        label_encoder.label_encoder.transform(['@unk'])[0])

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=parameters.embed_size, use_gpu=False)

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(csr)  # item-user 순으로 넣어야하는데 우리는 tags-songs이므로 item:tags, user:songs
    util.dump(model, './tags_songs_wmf.pickle')
