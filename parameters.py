embed_size = 100#100
multihead = 5
stack = 4

max_sequence_length = 107 # @song_cls || @tag_cls || songs(100) || tags(5)
title_max_sequence_length = 100

bpe_voca_size = 12000
bpe_character_coverage = 0.9995

val_test_max_songs = 100  # val, test에는 최대 100개 노래 존재
val_test_max_tags = 5  # val, test에는 최대 5개 태그 존재

songs_recall = 0.9
tags_recall = 0.95

base_dir = './'
song_issue_dict = 'song_issue_dict.pickle'
label_info = 'label_info.pickle'
bpe_model_prefix = 'plylst_title_bpe'
bpe_model_file = bpe_model_prefix + '.model'
plylst_titles = 'plylst_titles.txt'

songs_tags_wmf = 'songs_tags_wmf.pickle'
songs_tags_csr_matrix = 'songs_tags_csr_matrix.pickle'

plylst_songs_tags_wmf = 'plylst_songs_tags_wmf.pickle'
plylst_songs_tags_csr_matrix = 'plylst_songs_tags_csr_matrix.pickle'
