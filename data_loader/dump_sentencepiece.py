# https://github.com/google/sentencepiece/blob/master/python/README.md
# pip install sentencepiece

import util
import sentencepiece as spm


def dump_plylst_title(dataset, fout):
    with open(fout, 'w', encoding='utf-8', errors='ignore') as o:
        for each in dataset:
            plylst_title = each['plylst_title']
            o.write(plylst_title + '\n')

if __name__ == "__main__":
    train_set = util.load_json('dataset/orig/train.json')
    dump_plylst_title(train_set, fout='./train_plylst_title.txt')
    # spm.SentencePieceTrainer.train(input='test/botchan.txt', model_prefix='m', vocab_size=1000,
    #                                user_defined_symbols=['foo', 'bar'])