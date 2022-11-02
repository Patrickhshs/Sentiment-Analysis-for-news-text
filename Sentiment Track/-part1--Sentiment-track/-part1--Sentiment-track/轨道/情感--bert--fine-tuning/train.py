import datasets
from datasets import load_dataset
import torch
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import  BertTokenizer, LineByLineTextDataset, TrainingArguments, Trainer
from transformers import BertForSequenceClassification, AdamW
import math
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

class TrainMyBert():
    def __init__(self, train_path, dev_path, path, save_path):
        self.raw_dataset = load_dataset("csv", data_files={"train": train_path, "validation": dev_path})
        self.raw_train_dataset = self.raw_dataset["train"]
        self.raw_val_dataset = self.raw_dataset["validation"]
        self.model_path = path
        self.save_path = save_path
        self.init_model()
        self.dataset_process()


    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path,  # 小写的 12 层预训练模型
            num_labels=2,  # 分类数 --2 表示二分类
            # 你可以改变这个数字，用于多分类任务
            # output_attentions=False,  # 模型是否返回 attentions weights.
            # output_hidden_states=False,  # 模型是否返回所有隐层状态.
        )
        # 在 gpu 中运行该模型
        self.model.cuda()

    def dataset_process(self):
        self.raw_train_dataset = self.raw_train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

        self.raw_val_dataset = self.raw_val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

        MAX_LENGTH = 100
        self.train_dataset = self.raw_train_dataset.map(
            lambda e: self.tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.dev_dataset = self.raw_val_dataset.map(
            lambda e: self.tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)


    def model_train(self):

        #数据处理
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        self.dev_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        #定义评价函数
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        #设置训练参数
        training_args = TrainingArguments(
            output_dir=self.save_path,  # output directory
            learning_rate=5e-5,
            num_train_epochs=5,  # total number of training epochs
            per_device_train_batch_size=50,  # batch size per device during training
            #     per_device_eval_batch_size=5,   # batch size for evaluation
            #     logging_dir='./logs',            # directory for storing logs
            logging_steps=1,
            do_train=True,
            do_eval=True,
            no_cuda=False,
            # gradient_accumulation_steps=2,
            #     load_best_model_at_end=True,
            # eval_steps=100,
            #     evaluation_strategy="epoch"
        )

        #创建训练器
        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.dev_dataset,  # evaluation dataset
            compute_metrics=compute_metrics
        )

        #开始训练
        logging.info('Train Model start')
        self.trainer.train()

        logging.info("Trian end")

        #评估
        results = {}
        if training_args.do_eval:
            logging.info("*** Evaluate ***")

            eval_output = self.trainer.evaluate()
            print(eval_output)
            print("eval_loss: ", eval_output["eval_loss"])
            # perplexity = math.exp(eval_output["eval_loss"])
            # result = {"perplexity": perplexity}
            #
            # output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
            # # if trainer.is_world_master():
            # if self.trainer.is_world_process_zero():
            #     with open(output_eval_file, "w") as writer:
            #         logging.info("***** Eval results *****")
            #         for key in sorted(result.keys()):
            #             # logging.info(f"  {key} = {str(result[key])}")
            #             print("  %s = %s", key, str(result[key]))
            #             writer.write("%s = %s\n" % (key, str(result[key])))

            logging.info("*** Evaluate End***")

    def save_model(self):
        self.trainer.save_model()
        self.tokenizer.save_pretrained(save_directory=self.save_path)

def main(train_path, dev_path, path, save_path):

    my_model = TrainMyBert(train_path, dev_path, path, save_path)
    print('data: ', train_path)
    my_model.model_train()

    is_save = input("是否保存模型至 " + str(save_path))
    if is_save == 'true':
        my_model.save_model()


if __name__ == '__main__':
    train_path = 'data/train/biaoji_train7_0.csv'
    # train_path = 'data/4/train_0.csv'
    dev_path = 'data/train/dev.csv'
    # path = 'model/train1'
    path = 'result_6'
    save_path = 'result_6'
    main(train_path, dev_path, path, save_path)