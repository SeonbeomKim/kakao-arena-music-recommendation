import util
import numpy as np

class Label_info:
    def __init__(self, dataset, song_meta, songs_recall=0.9, tags_recall=0.95):
        self.songs = []
        self.tags = []
        self.cls_tokens = ['@song_cls', '@tag_cls']
        self.sep_token = '@sep'
        self.mask_token = '@mask'
        self.pad_token = '@pad'
        self.unk_token = '@unk'
        self.others_for_encoder = self.cls_tokens + [self.sep_token, self.mask_token,
                                   self.pad_token]  # TODO mask는 안쓰면 나중에 지우자.

        self.label_encoder = util.LabelEncoder(tokens=self.others_for_encoder, unk_token=self.unk_token)
        self.set_label_encoder(dataset, songs_recall=songs_recall, tags_recall=tags_recall)

        self.genre = []
        self.artist = []
        self.meta_label_encoder = util.LabelEncoder(tokens=self.others_for_encoder, unk_token=self.unk_token)
        self.set_meta_label_encoder(self.songs, song_meta)

        self.album = []
        self.album_label_encoder = util.LabelEncoder(tokens=self.others_for_encoder, unk_token=self.unk_token)
        self.set_album_label_encoder(self.songs, song_meta)


    def filter_row_freq_item(self, dataset, item_key, recall=0.9):
        freq_dict = {}

        total_freq = 0
        for each in dataset:
            for item in each[item_key]:
                if item not in freq_dict:
                    freq_dict[item] = 0
                freq_dict[item] += 1
                total_freq += 1

        print('%s num: %d:' % (item_key, len(freq_dict)))
        freq_sorted_items = sorted(list(freq_dict.items()), key=lambda each: each[1], reverse=True)

        accum_freq = 0
        slice_idx = 0
        for idx, item_freq in enumerate(freq_sorted_items):
            if (accum_freq / float(total_freq)) > recall:
                slice_idx = idx
                break
            song, freq = item_freq
            accum_freq += freq

        filtered_items = [item_freq[0] for item_freq in freq_sorted_items[:slice_idx]]
        filtered_freq = [item_freq[1] for item_freq in freq_sorted_items[:slice_idx]]
        print('%s num(after filtering): %d, freq: %d' % (
            item_key, len(filtered_items), freq_sorted_items[slice_idx - 1][1]))
        return filtered_items, filtered_freq

    def set_label_encoder(self, dataset, songs_recall=0.9, tags_recall=0.95):
        self.songs, songs_freq = self.filter_row_freq_item(dataset, 'songs', recall=songs_recall)
        self.tags, tags_freq = self.filter_row_freq_item(dataset, 'tags', recall=tags_recall)

        self.label_encoder.fit(self.songs + self.tags)

    def set_meta_label_encoder(self, songs, song_meta):
        songs_set = set(songs)
        genre_set = set()
        artist_set = set()

        for each in song_meta:
            if each['id'] not in songs_set:
                continue
            artist_set.update(each['artist_id_basket'])
            genre_set.update(each['song_gn_gnr_basket'])

        self.genre = list(genre_set)
        self.artist = list(artist_set)
        self.meta_label_encoder.fit(self.genre + self.artist)

    def set_album_label_encoder(self, songs, song_meta):
        songs_set = set(songs)
        album_set = set()

        for each in song_meta:
            if each['id'] not in songs_set:
                continue
            album_set.add(each['album_id'])

        self.album = list(album_set)
        self.album_label_encoder.fit(self.album)
