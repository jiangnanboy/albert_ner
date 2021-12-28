from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.data import BucketIterator

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GetDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, SPECIAL_TOKENS, label2i):
        '''
        build dataset
        :param data_path:
        :param tokenizer:
        :param max_length:
        :param SPECIAL_TOKENS:
        :param label2i:
        '''
        self.data_list = self.read_data(data_path)
        self.data_size = len(self.data_list)
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.max_length = max_length
        self.label2i = label2i

    def read_data(self, data_path):
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as data_read:
            words = []
            tags = []
            for line in data_read:
                line = line.strip()
                if line == '':
                    if len(words) != 0 and len(tags) != 0:
                        data_list.append([words, tags])
                        words = []
                        tags = []
                        continue
                line = line.split()
                words.append(line[0])
                tags.append(line[1])
            if len(words) !=0 and len(tags) != 0:
                data_list.append([words, tags])
        return data_list

    def __getitem__(self, idx):
        words, tags = self.data_list[idx]
        input = ''.join(words) # berttokenizer会在句子两端加入[cls]与[sep]
        n_pad = self.max_length - len(tags) # padding or truncation for label
        if n_pad <= 0:
            tags = tags[:self.max_length - 2]
            tags = [self.SPECIAL_TOKENS['cls_token']] + tags + [self.SPECIAL_TOKENS['sep_token']]
        elif n_pad == 1:
            tags = tags[:len(tags) - 1]
            tags = [self.SPECIAL_TOKENS['cls_token']] + tags+ [self.SPECIAL_TOKENS['sep_token']]
        else:
            tags = [self.SPECIAL_TOKENS['cls_token']] + tags + [self.SPECIAL_TOKENS['sep_token']]
            tags.extend([self.SPECIAL_TOKENS['pad_token']] * (self.max_length - len(tags)))
        label = [self.label2i[token] for token in tags]

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding='max_length')

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(label),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

    def __len__(self):
        return self.data_size

def get_train_val_dataloader(batch_size, trainset, train_ratio):
    '''
    split trainset to train and val
    :param batch_size:
    :param trainset:
    :param train_ratio
    :return:
    '''

    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    valloader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False)

    return trainloader, valloader, train_dataset, val_dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_iterator(dataset: Dataset, batch_size, sort_key=lambda x: len(x.input_ids), sort_within_batch=True, shuffle=True):
    return BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                          sort_within_batch=sort_within_batch, shuffle=shuffle)

def get_score(model, input_ids, attention_mask, label, score_type='f1'):
    metrics_map = {
        'f1': f1_score,
        'p': precision_score,
        'r': recall_score,
        'acc': accuracy_score
    }
    metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
    predict_y = model(input_idx=input_ids, attention_mask=attention_mask)[0]
    label = label[:len(predict_y)]

    assert len(label) == len(predict_y)
    return metric_func(predict_y, label, average='micro')