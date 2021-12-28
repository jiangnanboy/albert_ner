# albert-crf

## 概述

利用huggingface/transformers中的albert+crf进行中文实体识别

利用albert加载中文预训练模型，后接一个前馈分类网络，最后接一层crf。利用albert预训练模型进行fine-tune。

整个流程是：

- 数据经albert后获取最后的隐层hidden_state=768
- hidden_state=768经一层前馈网络进行分类
- 将前馈网络的分类结果输入crf

 ## 数据说明

BIO形式标注（来自人民日报），见data/

训练数据示例如下：

```
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O
```

## 训练和预测见（examples/test_ner.py）

```
ner = NER(args)
if train_bool(args.train):
    ner.train()
else:
    ner.load()
    # ner.test(args.test_path)
    pprint(ner.predict('据新华社报道，安徽省六安市被评上十大易居城市！'))
    #[{'start': 7, 'stop': 10, 'type': 'LOC', 'word': '安徽省'},{'start': 10, 'stop': 13, 'type': 'LOC', 'word': '六安市'},{'start': 1, 'stop': 4, 'type': 'ORG', 'word': '新华社'}]
    
    print(ner.predict('相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。'))
    #[{'start': 5, 'stop': 10, 'word': '青岛海牛队', 'type': 'ORG'}, {'start': 11, 'stop': 16, 'word': '广州松日队', 'type': 'ORG'}]
    
```

## 项目结构
- data
    - example.dev
    - example.train
- examples
    - test_ner.py #训练及预测
- model
    - pretrained_model #存放预训练模型和相关配置文件
        - config.json
        - pytorch_model.bin
        - vocab.txt
- ner
    - utils
        - convert.py
    - dataset.py
    - model.py
    - module.py
- utils
    - log.py

## 参考
- [transformers](https://github.com/huggingface/transformers)