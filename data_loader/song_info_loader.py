# coding=utf-8
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from util import load_json


class SongMetaInfoLoader(object):
    def __init__(self):
        self.song_tag_map = {}
        self.song_genre_map = {}
        self.song_detail_genre_map = {}
        self.song_album_id_map = {}
        self.song_album_name_map = {}
        self.song_artist_map = {}

    def _make_song_tag_info(self, playlist_data, top_n):
        song_tag_dict = defaultdict(list)
        for playlist in tqdm(playlist_data):
            tag = playlist['tags']
            songs = playlist['songs']

            for song_id in songs:
                song_tag_dict[song_id].extend(tag)

        for song_id, tag_list in tqdm(song_tag_dict.items()):
            tag_counter = Counter(tag_list).most_common(top_n)
            self.song_tag_map[song_id] = list(map(lambda kv: kv[0], tag_counter))

    def _make_song_metainfo_map(self, meta_data, top_n):
        for meta in tqdm(meta_data):
            song_id = meta['id']
            self.song_artist_map[song_id] = meta['artist_id_basket'][:top_n]
            self.song_album_id_map[song_id] = meta['album_id']
            self.song_album_name_map[song_id] = meta['album_name']

            genre_list = meta['song_gn_gnr_basket'][:top_n]
            dup_genres = set(map(lambda g: str(g)[:-1]+"1", genre_list))
            detail_genre = set(meta['song_gn_dtl_gnr_basket']) - dup_genres

            self.song_genre_map[song_id] = genre_list
            self.song_detail_genre_map[song_id] = detail_genre

    def make_info_map(self, playlist_data, meta_data, top_n=5):
        print('make meta info of songs!')
        self._make_song_tag_info(playlist_data=playlist_data, top_n=top_n)
        self._make_song_metainfo_map(meta_data=meta_data, top_n=5)
        print('done!')


if __name__ == '__main__':
    playlist_info = load_json('../res/orig/train.json')
    meta_info = load_json('../res/song_meta.json')
    meta_info_loader = SongMetaInfoLoader()
    meta_info_loader.make_info_map(playlist_data=playlist_info, meta_data=meta_info)

    test_song_id = 17
    print(meta_info_loader.song_tag_map[test_song_id])
    print(meta_info_loader.song_genre_map[test_song_id])
    print(meta_info_loader.song_detail_genre_map[test_song_id])
    print(meta_info_loader.song_album_id_map[test_song_id])
    print(meta_info_loader.song_album_name_map[test_song_id])
    print(meta_info_loader.song_artist_map[test_song_id])
