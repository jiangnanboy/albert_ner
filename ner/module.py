import torch
from tqdm import tqdm
import random
import os
import numpy as np

import sys
sys.path.append('/home/sy/project/albert_crf/')

from utils.log import logger

from ner.model import AlbertCrf, load_tokenizer, load_config, load_pretrained_model, build_model

from ner.dataset import GetDataset, get_dataloader, get_score
from ner.utils.convert import get_tags, format_result

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    set seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2021)

class NER():
    '''
    ner
    '''
    def __init__(self, args):
        self.args = args
        self.SPECIAL_TOKEN = args.SPECIAL_TOKEN
        self.label2i = args.LABEL2I
        self.model = None
        self.tokenizer = None

    def train(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        pretrained_model, albertConfig = load_pretrained_model(self.args.pretrained_model_path, self.tokenizer, self.SPECIAL_TOKEN)

        train_set = GetDataset(self.args.train_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)

        if self.args.dev_path:
            dev_dataset = GetDataset(self.args.dev_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)
            val_iter = get_dataloader(dev_dataset, batch_size=self.args.batch_size, shuffle=False)
        train_iter = get_dataloader(train_set, batch_size=self.args.batch_size)

        tag_num = len(self.label2i)
        albertcrf = AlbertCrf(albertConfig, pretrained_model, tag_num)
        self.model = albertcrf.to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.1)

        best_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            self.model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self.model.zero_grad()
                label = item['label']
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                item_loss = (-self.model.loss(input_idx=input_ids, attention_mask=attention_mask, labels=label)) / label.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))

            if self.args.dev_path:
                val_loss = self.validate(dev_dataset=dev_dataset, val_iter=val_iter)
                # val_loss = self._validate(val_iter=val_iter)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save model
                    torch.save(self.model.state_dict(), self.args.model_path)
                    # torch.save(self.model, self.args.model_path)
                    logger.info('save model : {}'.format(self.args.model_path))
                logger.info('val_loss: {}, best_val_loss: {}'.format(val_loss, best_val_loss))

            # scheduler.step()

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            # input_text = self.SPECIAL_TOKEN['cls_token'] + text + self.SPECIAL_TOKEN['sep_token']
            # input_text_encoder = torch.tensor(self.tokenizer.encode(input_text)).unsqueeze(0)
            encodings_dict = self.tokenizer(text,
                                            truncation=True,
                                            max_length=self.args.max_length,
                                            padding='max_length')

            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            vec_predict = self.model(input_idx=input_ids, attention_mask=attention_mask)[0]

            '''
            i2label = {value:key for key,value in self.label2i.items()}
            tag_predict = [i2label[i] for i in vec_predict]
            tag_predict = tag_predict[1:len(tag_predict)-1] # except 'cls' and 'sep'
            print('text: {}'.format(text))
            print('tag_predict: {}'.format(tag_predict))
            return iob_ranges(list(text), tag_predict)
            '''

            tags = set()
            for label_tag in self.label2i:
                if '-' in label_tag:
                    tag = label_tag.split('-')[1]
                    tags.add(tag)

            entities = []
            for tag in tags:
                ner_tags = get_tags(vec_predict[1:len(vec_predict)-1], tag, self.label2i)
                entities += format_result(ner_tags, text, tag)
            return entities

    def load(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        albertConfig = load_config(self.args.pretrained_model_path, self.tokenizer)
        albert_model = build_model(albertConfig)
        tag_num = len(self.label2i)
        self.model = AlbertCrf(albertConfig, albert_model, tag_num)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=DEVICE))
        logger.info('loading model {}'.format(self.args.model_path))
        self.model = self.model.to(DEVICE)

    def test(self, test_path):
        test_dataset = GetDataset(test_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)
        test_score = self._validate(test_dataset)
        return test_score

    def _validate(self, test_dataset):
        self.model.eval()
        with torch.no_grad():
            test_score_list = []
            for dev_item in tqdm(test_dataset):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['attention_mask']

                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                item_score = get_score(model=self.model, input_ids=input_ids.unsqueeze(0),
                                       attention_mask=attention_mask.unsqueeze(0), label=label)

                test_score_list.append(item_score)
            print('test dataset len:{}'.format(len(test_score_list)))
            test_score = sum(test_score_list) / len(test_score_list)
            logger.info('test_score: {}'.format(test_score))
        return test_score

    def validate(self, dev_dataset, val_iter):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for dev_item in tqdm(val_iter):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['attention_mask']
                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                print('label:{}'.format(label.shape))
                print('input_ids:{}'.format(input_ids.shape))

                item_loss = (-self.model.loss(input_idx=input_ids, attention_mask=attention_mask,
                                              labels=label)) / label.size(1)
                val_loss += item_loss.view(-1).cpu().data.tolist()[0]

            dev_score_list = []
            for dev_item in tqdm(dev_dataset):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['attention_mask']

                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                item_score = get_score(model=self.model, input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), label=label)

                dev_score_list.append(item_score)
            print('dev dataset len:{}'.format(len(dev_score_list)))
            logger.info('dev_score: {}'.format(sum(dev_score_list) / len(dev_score_list)))
        return val_loss
