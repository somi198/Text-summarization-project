import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KoBARTSummaryDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from rouge import Rouge

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--checkpoint_path',            # 모델 체크포인트 경로 입력
                    type=str,
                    help='checkpoint path')

parser.add_argument('--mode',
                    type=str, help='train/test')    # train 입력 시 train만 진행, test 입력 시 test만 진행

parser.add_argument('--hparams_file',               # 하이퍼파라미터 정보가 담긴 yaml 파일 경로 입력
                    type=str, help='input hparams file path')


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train.tsv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=16,
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class KobartSummaryModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=8,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        if tok is None:
            self.tok = get_kobart_tokenizer()
        else:
            self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        # num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)  
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
        
        
    def test_step(self, batch, batch_idx):
        score = {'rouge-1':{'r':0, 'p':0, 'f':0}, 
                 'rouge-2':{'r':0, 'p':0, 'f':0}, 
                 'rouge-l':{'r':0, 'p':0, 'f':0}}
        rouge = Rouge()
        
        x = batch['input_ids']
        y = batch['label']

        output = self.model.generate(x, eos_token_id=1, max_length=512, num_beams=5)
        
        
        for i in range(len(output)):

            predict = self.tokenizer.decode(output[i], skip_special_tokens=True)
            label = y[i]
            
            s = rouge.get_scores(predict, label)[0]
            
            
            score['rouge-1']['r'] += s['rouge-1']['r']
            score['rouge-1']['p'] += s['rouge-1']['p']
            score['rouge-1']['f'] += s['rouge-1']['f']
            
            score['rouge-2']['r'] += s['rouge-2']['r']
            score['rouge-2']['p'] += s['rouge-2']['p']
            score['rouge-2']['f'] += s['rouge-2']['f']
            
            score['rouge-l']['r'] += s['rouge-l']['r']
            score['rouge-l']['p'] += s['rouge-l']['p']
            score['rouge-l']['f'] += s['rouge-l']['f']
            
            
        score['rouge-1']['r'] /= len(output)
        score['rouge-1']['p'] /= len(output)
        score['rouge-1']['f'] /= len(output)
            
        score['rouge-2']['r'] /= len(output)
        score['rouge-2']['p'] /= len(output)
        score['rouge-2']['f'] /= len(output)
            
        score['rouge-l']['r'] /= len(output)
        score['rouge-l']['p'] /= len(output)
        score['rouge-l']['f'] /= len(output)
        
        #print(score)
           
        return (score)
    
    
    
    def test_epoch_end(self, outputs):
        score = {'rouge-1':{'r':0, 'p':0, 'f':0}, 'rouge-2':{'r':0, 'p':0, 'f':0}, 'rouge-l':{'r':0, 'p':0, 'f':0}}
        
        for s in outputs:
            score['rouge-1']['r'] += s['rouge-1']['r']
            score['rouge-1']['p'] += s['rouge-1']['p']
            score['rouge-1']['f'] += s['rouge-1']['f']
            
            score['rouge-2']['r'] += s['rouge-2']['r']
            score['rouge-2']['p'] += s['rouge-2']['p']
            score['rouge-2']['f'] += s['rouge-2']['f']
            
            score['rouge-l']['r'] += s['rouge-l']['r']
            score['rouge-l']['p'] += s['rouge-l']['p']
            score['rouge-l']['f'] += s['rouge-l']['f']
            
        
        score['rouge-1']['r'] /= len(outputs)
        score['rouge-1']['p'] /= len(outputs)
        score['rouge-1']['f'] /= len(outputs)
            
        score['rouge-2']['r'] /= len(outputs)
        score['rouge-2']['p'] /= len(outputs)
        score['rouge-2']['f'] /= len(outputs)
            
        score['rouge-l']['r'] /= len(outputs)
        score['rouge-l']['p'] /= len(outputs)
        score['rouge-l']['f'] /= len(outputs)
        
        
        df = pd.DataFrame(score)
        df.to_csv(os.path.join(args.default_root_dir, 'rouge_score.csv'))
        
        print(df)
        
        
        
if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)
    
    dm = KobartSummaryModule(args.train_file,
                            args.test_file,
                            None,
                            batch_size=args.batch_size,
                            max_len=args.max_len,
                            num_workers=args.num_workers)
    
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                               dirpath=args.default_root_dir,
                                               filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                               verbose=True,
                                               save_last=True,
                                               mode='min',
                                               save_top_k=-1,
                                               prefix='kobart_summary')

        
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,   # metric 성능이 몇 번의 epoch가 향상 되지않을 때 학습을 멈출건지 지정
        verbose=True,
        mode='min'
    )
        
        
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, auto_scale_batch_size='power', logger=tb_logger,
                                            callbacks=[checkpoint_callback, early_stopping, lr_logger])
       
    
    if args.mode == 'train':       
            
        if not args.checkpoint_path and not args.hparams_file:   # 체크포인트 경로나 yaml 파일 경로가 입력되지 않았을 경우
            model = KoBARTConditionalGeneration(args)            # 모델 새로 생성
        else:                                                    # 체크포인트 경로와 yaml 파일 경로 모두 입력된 경우
            with open(args.hparams_file) as f:                   
                hparams = yaml.load(f, Loader=yaml.Loader)
            
            model = KoBARTConditionalGeneration.load_from_checkpoint(args.checkpoint_path, hparams=hparams) # 해당 모델 로드
       
        
        '''
        <lr finder 사용 시>
         lr_finder = trainer.tuner.lr_find(model=model, datamodule=dm)
         lr_finder.results
         new_lr = lr_finder.suggestion() # Pick point based on plot, or get suggestion
         model.hparams.lr = new_lr
        '''

        #trainer.tune(model,dm)  # 최적의 batch size 찾을 시
        trainer.fit(model, dm)
        
        
    elif args.mode == 'test':
        
        with open(args.hparams_file) as f:
            hparams = yaml.load(f, Loader=yaml.Loader)
            
        model=KoBARTConditionalGeneration.load_from_checkpoint(args.checkpoint_path, hparams=hparams)
       

        # test (pass in the model)
        trainer.test(model=model, datamodule=dm, verbose=True)
                
