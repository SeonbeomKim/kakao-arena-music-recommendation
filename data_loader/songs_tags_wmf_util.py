from scipy.sparse import csr_matrix
from tqdm import tqdm

def make_sparse_matrix(dataset, label_encoder, songs_num, unk_idx):
    tags_songs_dict = {}
    for each in tqdm(dataset, total=len(dataset)):
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
