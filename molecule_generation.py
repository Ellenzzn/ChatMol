import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os
from torch.utils import data
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
import joblib
import numpy as np
import argparse
import random
from torch.nn.parallel import DataParallel
from tqdm import tqdm, trange
import torch.nn as nn
import nltk
from nltk.translate.bleu_score import SmoothingFunction


class QA_dataset(Dataset):
    def __init__(self, tokenizer, pth_i, pth_o, max_len=512):  
        self.data = (open(pth_i)).readlines()  
        self.ans = (open(pth_o)).readlines()  
        self.len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp = self.tokenizer.encode(self.data[index].strip('\n'))
        if len(inp)>512:         
            inp=inp[-512:]
        out = self.tokenizer.encode(self.ans[index].strip('\n'))
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)  
        labels = torch.ones(self.len) * (-100)
        inp_ids[:min(self.len, len(inp))] = torch.from_numpy(np.array(inp))[:min(self.len, len(inp))]
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.from_numpy(np.array(out))[:min(self.len, len(out))]

        return inp_ids.long(), att_msk.long(), labels.long()


def do_eval(tokenizer, model, dev_dataloader):
    cnt = 0
    bleu2 = 0
    bleu4 = 0
    nist = 0
    acc = 0
    acc3=0
    f1 = open('inp_base.txt', 'w')
    f2 = open('pred_base.txt', 'w')
    f3 = open('ans_base.txt', 'w')
    rec = []
    with torch.no_grad():
        for idx, i in enumerate(tqdm(dev_dataloader)):  
            inputs, atts, labels = i

            output = model.generate(input_ids=inputs.cuda(), attention_mask=atts.cuda(), max_length=512,
                                           num_beams=3, num_return_sequences=3)
            tmp = (labels > 0) * labels + (labels < 0)

            inp = [tokenizer.decode(piece).split('</s>')[0] for piece in inputs]
            pred = [tokenizer.decode(piece).split('</s>')[0].strip('<pad> ') for piece in output]
            ans = [tokenizer.decode(piece).split('</s>')[0] for piece in tmp]

            for st in inp:
                f1.write(st + '\n')
                f1.flush()  
            for st in pred:
                f2.write(st + '\n')
                f2.flush()
            for st in ans:
                f3.write(st + '\n')
                f3.flush()
            for one in range(len(ans)):
                if pred[3 * one] == ans[one]:
                    acc += 1
                
                if ans[one] in pred[3*one:3*one+2]:
                    acc3+=1
                if sum(output[one] > 1) < 4:
                    cnt += 1
                    continue
                pred1 = [pred[3 * one].strip('\n')]
                gt1 = [[ans[one].strip('\n')]]
                try:
                    bleu2 += nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 1, 0, 0))
                except ZeroDivisionError:
                    0
                try:
                    bleu4 += nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 0, 0, 1))
                except ZeroDivisionError:
                    0
                try:
                    nist += nltk.translate.nist_score.corpus_nist(gt1, pred1, n=4)
                except ZeroDivisionError:
                    0
                rec.append([wd for wd in pred1[0]])
                cnt += 1
            print(acc / cnt, acc3/cnt,  bleu2 / cnt, bleu4 / cnt, nist / cnt)
    return acc / cnt


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    config = T5Config(
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=1,  
        decoder_start_token_id=1,
        eos_token_id=2,
        gradient_checkpointing=False)

    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    
    if args.init_checkpoint is not None:
        pt = torch.load(args.init_checkpoint).state_dict()#['state_dict']
        pt1 = {k[7:]: v for k, v in pt.items()}
        model.load_state_dict(pt1, strict=True)
    
    pt = None
    pt1 = None

    model = model.cuda()
    #model = torch.nn.parallel.DataParallel(model)  # 并行

    params = []
    for n, p in model.named_parameters()
        params.append(p)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    global_step = 0
    if args.resume > -1:
        pt = torch.load(str(args.resume)).state_dict()  
        model.load_state_dict(pt, strict=True)
    train_dataset = QA_dataset(tokenizer, args.pth_train + 'inp.txt', args.pth_train + 'out.txt')
    dev_dataset = QA_dataset(tokenizer, args.pth_dev + 'inp.txt', args.pth_dev + 'out.txt')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=True,
                                  num_workers=4, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, 
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True)
    avg_loss = 0  
    best_acc = 0
    for epoch in range(args.epoch):
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            inputs, atts, labels = i    
            output = model(input_ids=inputs.cuda(), attention_mask=atts.cuda(), labels=labels.cuda(), return_dict=True)
            loss = output.loss.mean() / 2  #
            loss.backward()
            avg_loss += loss.item()
            if idx % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    print('Step: ', global_step, ', loss: ', avg_loss)
                    avg_loss = 0

                if global_step > args.global_step:   
                    return
                
                if global_step % 5000 == 0:  
                    if global_step > args.step_pre:
                    torch.save(model, args.save_pth + str(global_step) + '.pt')


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default='save_model/preT2M_base.pt', type=str, )
    parser.add_argument("--save_pth", default='save_model/dia_pre_base_', type=str, )
    parser.add_argument("--lr", default=5e-4, type=float, )
    parser.add_argument("--resume", default=-1, type=int, )
    parser.add_argument("--pth_train", default='data/ChEBI-dia/train_', type=str, )
    parser.add_argument("--pth_dev", default='data/ChEBI-dia/dev_', type=str, )
    parser.add_argument("--batch_size", default=8, type=int, )  # 32
    parser.add_argument("--epoch", default=100, type=int, )
    parser.add_argument("--seed", default=1234, type=int, )
    parser.add_argument("--global_step", default=100001, type=int, )
    parser.add_argument("--step_pre", default=-1, type=int, )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
