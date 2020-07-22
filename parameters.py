songs_cls_token = '@songs_cls'
tags_cls_token = '@tags_cls'
artists_cls_token = '@artists_cls'
sep_token = '@sep'
mask_token = '@mask'
pad_token = '@pad'

songs_tags_artists_model_embed_size = 240
songs_tags_artists_model_multihead = 2#2
songs_tags_artists_model_stack = 1
songs_tags_artists_model_max_sequence_length = 280

title_model_embed_size = 132
title_model_multihead = 2#2
title_model_stack = 2
title_model_max_sequence_length = 53

bpe_voca_size = 6000
bpe_character_coverage = 0.9995

val_test_max_songs = 100  # val, test에는 최대 100개 노래 존재
val_test_max_tags = 5  # val, test에는 최대 5개 태그 존재

songs_recall = 0.9
tags_recall = 0.95
artists_recall = 0.98

base_dir = './'
song_issue_dict = 'song_issue_dict.pickle'
label_info = 'label_info.pickle'
bpe_model_prefix = 'plylst_title_bpe'
bpe_model_file = bpe_model_prefix + '.model'
plylst_titles = 'plylst_titles.txt'
