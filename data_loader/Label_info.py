import parameters

import util


class Label_info:
    def __init__(self, dataset, song_meta, songs_recall=0.9, tags_recall=0.95, artists_recall=0.9):
        self.songs = []
        self.tags = []
        self.others_for_encoder = [parameters.songs_cls_token, parameters.tags_cls_token, parameters.artists_cls_token,
                                   parameters.pad_token, parameters.sep_token]

        self.label_encoder = util.LabelEncoder(tokens=self.others_for_encoder)
        self.set_label_encoder(dataset, song_meta, songs_recall=songs_recall, tags_recall=tags_recall,
                               artists_recall=artists_recall)

    def get_item_freq_dict(self, dataset, item_key):
        freq_dict = {}

        for each in dataset:
            for item in each[item_key]:
                if item not in freq_dict:
                    freq_dict[item] = 0
                freq_dict[item] += 1

        print('%s num: %d:' % (item_key, len(freq_dict)))
        return freq_dict

    def get_artists_freq_dict(self, dataset, song_meta, all_songs_set):
        song_artist_dict = self.get_song_artists_meta_dict(song_meta, all_songs_set)
        freq_dict = {}

        for each in dataset:
            songs = list(filter(lambda song: song in all_songs_set, each['songs']))
            artists_set = set()
            for song in songs:
                artists_set.update(song_artist_dict[song])

            # plylst마다 unique하게 몇번 등장하는지 확인함
            for artist in artists_set:
                if artist not in freq_dict:
                    freq_dict[artist] = 0
                freq_dict[artist] += 1

        print('%s num: %d:' % ('artist', len(freq_dict)))
        return freq_dict

    def get_song_artists_meta_dict(self, song_meta, all_songs_set, all_key_data_set=set()):
        song_artist_dict = {}
        for each in song_meta:
            song = each['id']
            if song not in all_songs_set:
                continue

            artist = list(map(lambda x: 'artist_%d' % x, each['artist_id_basket']))
            if all_key_data_set:
                artist = list(filter(lambda x: x in all_key_data_set, artist))

            if not artist:
                continue

            song_artist_dict[song] = artist
        return song_artist_dict

    def filter_low_freq_item(self, freq_dict, recall=0.9):
        total_freq = sum(freq_dict.values())
        freq_sorted_items = sorted(list(freq_dict.items()), key=lambda each: each[1], reverse=True)

        if float(recall) == 1.0:
            slice_idx = len(freq_sorted_items)
        else:
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
        print('num(after filtering): %d, freq: %d' % (len(filtered_items), freq_sorted_items[slice_idx - 1][1]))
        return filtered_items, filtered_freq

    def set_label_encoder(self, dataset, song_meta, songs_recall, tags_recall, artists_recall):
        songs_freq_dict = self.get_item_freq_dict(dataset, 'songs')
        self.songs, songs_freq = self.filter_low_freq_item(songs_freq_dict, recall=songs_recall)

        tags_freq_dict = self.get_item_freq_dict(dataset, 'tags')
        self.tags, tags_freq = self.filter_low_freq_item(tags_freq_dict, recall=tags_recall)

        artists_freq_dict = self.get_artists_freq_dict(dataset, song_meta, set(self.songs))
        self.artists, artists_freq = self.filter_low_freq_item(artists_freq_dict, recall=artists_recall)

        self.song_artist_dict = self.get_song_artists_meta_dict(song_meta, set(self.songs), set(self.artists))
        print('len(song_artist_dict):', len(self.song_artist_dict))
        self.label_encoder.fit(self.songs + self.tags + self.artists)
