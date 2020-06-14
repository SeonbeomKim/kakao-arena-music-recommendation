import os

import sentencepiece as spm

import data_loader.plylst_title_util as plylst_title_util
import parameters
import util
from data_loader.Label_info import Label_info

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    val_question = util.load_json('dataset/questions/val.json')
    val_answers = util.load_json('dataset/answers/val.json')
    song_meta = util.load_json('dataset/song_meta.json')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set, song_meta)
        util.dump(label_info, label_info_path)

    # bpe
    model_file = os.path.join(parameters.base_dir, parameters.bpe_model_file)
    if not os.path.exists(model_file):
        plylst_title_util.dump_plylst_title(train_set, fout=os.path.join(parameters.base_dir, parameters.plylst_titles))

        # sentencepiece
        spm.SentencePieceTrainer.train(
            input=os.path.join(parameters.base_dir, parameters.plylst_titles),
            model_prefix=parameters.bpe_model_prefix,
            vocab_size=parameters.bpe_voca_size,
            character_coverage=parameters.bpe_character_coverage,
            model_type='bpe',
            user_defined_symbols=['@cls', '@sep', '@mask', '@pad', '@unk'])

    sp = spm.SentencePieceProcessor(model_file=model_file)

    val_util = plylst_title_util.ValPlylstTitleUtil(val_question, val_answers, song_meta,
                                                    parameters.title_max_sequence_length, label_info, sp)
    util.dump(val_util, os.path.join(parameters.base_dir, parameters.plylst_title_transformer_val_sampled_data))
