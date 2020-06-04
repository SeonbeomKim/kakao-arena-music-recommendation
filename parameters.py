embed_size = 128 # multihead의 배수여야 함
multihead = 4
stack = 4
max_sequence_length = 108 # cls || songs(100) || sep || tags(5) || sep
title_max_sequence_length = 100

bpe_voca_size = 12000
bpe_character_coverage = 0.9995

input_bucket_size = [54, 108]
output_bucket_size = [105, 210]

val_test_max_songs = 100  # val, test에는 최대 100개 노래 존재
val_test_max_tags = 5  # val, test에는 최대 5개 태그 존재

base_dir = './'
label_info = 'label_info.pickle'

songs_tags_wmf = 'songs_tags_wmf.pickle'
songs_tags_csr_matrix = 'songs_tags_csr_matrix.pickle'

plylst_songs_tags_wmf = 'plylst_songs_tags_wmf.pickle'
plylst_songs_tags_csr_matrix = 'plylst_songs_tags_csr_matrix.pickle'

songs_tags_transformer_train_input_output = 'songs_tags_transformer_train_input_output.pickle'
songs_tags_transformer_val_input_output = 'songs_tags_transformer_val_input_output.pickle'
songs_tags_transformer_val_sampled_data = 'songs_tags_transformer_val_sampled_data.pickle'

bpe_model_prefix = 'plylst_title_bpe'
bpe_model_file = bpe_model_prefix + '.model'
plylst_titles = 'plylst_titles.txt'

plylst_title_transformer_train_input_output = 'plylst_title_transformer_train_input_output.pickle'
plylst_title_transformer_val_input_output = 'plylst_title_transformer_val_input_output.pickle'
plylst_title_transformer_val_sampled_data = 'plylst_title_transformer_val_sampled_data.pickle'