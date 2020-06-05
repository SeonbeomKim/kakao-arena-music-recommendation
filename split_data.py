# -*- coding: utf-8 -*-
import copy
import fire
import numpy as np
import os
import random

from util import load_json, write_json

np.random.seed(777)
random.seed(777)


class ArenaSplitter:
    def _split_data(self, playlists, ratio=0.8):
        tot = len(playlists)
        train = playlists[:int(tot * ratio)]
        val = playlists[int(tot * ratio):]

        return train, val

    def _mask(self, playlists, mask_cols, del_cols):
        q_pl = copy.deepcopy(playlists)
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            for del_col in del_cols:
                q_pl[i][del_col] = []
                if del_col == 'songs':
                    a_pl[i][del_col] = a_pl[i][del_col][:100]
                elif del_col == 'tags':
                    a_pl[i][del_col] = a_pl[i][del_col][:10]

            for col in mask_cols:
                mask_len = len(playlists[i][col])
                mask = np.full(mask_len, False)
                mask[:mask_len // 2] = True
                np.random.shuffle(mask)

                q_pl[i][col] = list(np.array(q_pl[i][col])[mask])
                a_pl[i][col] = list(np.array(a_pl[i][col])[np.invert(mask)])

        return q_pl, a_pl

    def _mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists[:int(tot * 0.3)]
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print('total: %d, Song only: %d, Song & Tags: %d, Tags only: %d, Title only: %d' % (
            len(playlists), len(song_only), len(song_and_tags), len(tags_only), len(title_only)))
        # print(f"Total: {len(playlists)}, "
        #       f"Song only: {len(song_only)}, "
        #       f"Song & Tags: {len(song_and_tags)}, "
        #       f"Tags only: {len(tags_only)}, "
        #       f"Title only: {len(title_only)}")

        song_q, song_a = self._mask(song_only, ['songs'], ['tags'])
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], [])
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs'])
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags'])

        q = song_q + songtag_q + tag_q + title_q
        a = song_a + songtag_a + tag_a + title_a

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])
        a = list(np.array(a)[shuffle_indices])

        return q, a

    def run(self, fname):

        print("Reading data...\n")
        playlists = load_json(fname)
        random.shuffle(playlists)
        print("Total playlists: %d" % len(playlists))
        # print(f"Total playlists: {len(playlists)}")

        print("Splitting data...")
        train, val = self._split_data(playlists)

        parent = os.path.dirname(fname)

        print("Original train...")
        write_json(train, os.path.join(parent, "orig/train.json"))
        # print("Original val...")
        # write_json(val, os.path.join(parent, "orig/val.json"))

        val, test = self._split_data(val, ratio=0.7)
        print("Original val...")
        write_json(val, os.path.join(parent, "orig/val.json"))

        #
        # print("Masked val...")
        # val_q, val_a = self._mask_data(val)
        # write_json(val_q, os.path.join(parent, "questions/val.json"))
        # write_json(val_a, os.path.join(parent, "answers/val.json"))


        print("Masked test...")
        test_q, test_a = self._mask_data(test)
        write_json(test_q, os.path.join(parent, "questions/test.json"))
        write_json(test_a, os.path.join(parent, "answers/test.json"))


if __name__ == "__main__":
    fire.Fire(ArenaSplitter)
