import os

import sentencepiece as spm

import data_loader.plylst_title_util as plylst_title_util
import parameters
import util
from data_loader.Label_info import Label_info

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    val_set = util.load_json('dataset/orig/val.json')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set)
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

    train_input_output = plylst_title_util.make_model_input_output(train_set, label_info)
    util.dump(train_input_output,
              os.path.join(parameters.base_dir, parameters.plylst_title_transformer_train_input_output))

    val_input_output = plylst_title_util.make_model_input_output(val_set, label_info)
    util.dump(val_input_output, os.path.join(parameters.base_dir, parameters.plylst_title_transformer_val_input_output))

    # validation을 위해 val set은 미리 고정해서 저장.
    model_val_dataset = plylst_title_util.make_train_val_set(val_input_output, parameters.title_max_sequence_length,
                                                             label_info=label_info, sentencepiece=sp, sample=30,
                                                             shuffle=False)
    util.dump(model_val_dataset,
              os.path.join(parameters.base_dir, parameters.plylst_title_transformer_val_sampled_data))
