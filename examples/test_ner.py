import os
import argparse
from pprint import pprint

import sys
sys.path.append('/home/sy/project/albert_crf/')

from ner.module import NER

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("Base path : {}".format(path))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(path, 'model/pytorch_model.bin'),
        type=str,
        required=False,
        help="The path of model!",
    )
    parser.add_argument(
        '--SPECIAL_TOKEN',
        default={"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"},
        type=dict,
        required=False,
        help='The dictionary of special tokens!'
    )
    parser.add_argument(
        '--LABEL2I',
        default={"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3,
                 "O": 4, "B-PER": 5, "I-PER": 6,
                 "B-LOC": 7, "I-LOC": 8,
                 "B-ORG": 9, "I-ORG": 10},
        type=dict,
        required=False,
        help='The dictionary of label2i!'
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join(path, 'data/example.train'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument(
        '--dev_path',
        default=os.path.join(path, 'data/example.dev'),
        type=str,
        required=False,
        help='The path of dev set!'
    )
    parser.add_argument(
        '--test_path',
        default=None,
        type=str,
        required=False,
        help='The path of test set!'
    )
    parser.add_argument(
        '--log_path',
        default=None,
        type=str,
        required=False,
        help='The path of Log!'
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=32, type=int, required=False, help="Batch size!"
    )
    parser.add_argument('--step_size', default=30, type=int, required=False, help='lr_scheduler step size!')
    parser.add_argument("--lr", default=0.001, type=float, required=False, help="Learning rate!")
    parser.add_argument('--clip', default=5, type=float, required=False, help='Clip!')
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=300, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument('--train', default='flase', type=str, required=False, help='Train or predict!')
    args = parser.parse_args()
    train_bool = lambda x:x.lower() == 'true'
    ner = NER(args)
    if train_bool(args.train):
        ner.train()
    else:
        ner.load()
        # ner.test(args.test_path)
        pprint(ner.predict('据新华社报道，安徽省六安市被评上十大易居城市！'))
        print(ner.predict('相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。'))



