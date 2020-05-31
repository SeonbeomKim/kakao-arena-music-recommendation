import os

import data_loader.songs_tags_util as songs_tags_util
from data_loader.Label_info import Label_info
import parameters
import util

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    val_set = util.load_json('dataset/orig/val.json')

    label_info_path = os.path.join(parameters.base_dir, parameters.label_info)
    if os.path.exists(label_info_path):
        label_info = util.load(label_info_path)
    else:
        label_info = Label_info(train_set)
        util.dump(label_info, label_info_path)

    train_input_output = songs_tags_util.make_model_input_output(train_set, label_info)
    util.dump(train_input_output,
              os.path.join(parameters.base_dir, parameters.songs_tags_transformer_train_input_output))

    val_input_output = songs_tags_util.make_model_input_output(val_set, label_info)
    util.dump(val_input_output, os.path.join(parameters.base_dir, parameters.songs_tags_transformer_val_input_output))

    # validation을 위해 val set은 미리 고정해서 저장.
    model_val_dataset = songs_tags_util.make_train_val_set(val_input_output, parameters.input_bucket_size,
                                                           parameters.output_bucket_size, sample=50,
                                                           label_info=label_info, shuffle=False)
    util.dump(model_val_dataset, os.path.join(parameters.base_dir, parameters.songs_tags_transformer_val_sampled_data))
