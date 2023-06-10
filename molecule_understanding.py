import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
import joblib
import numpy as np
import argparse
import random
from torch.nn.parallel import DataParallel
from tqdm import tqdm, trange
import torch.nn as nn
import pdb
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric,text_translation_metrics
import pandas as pd

class convert_dataset(Dataset):
    def __init__(self, pth, tokenizer, tag, max_length=256, few=1):
        dict_new = pd.read_csv(pth, sep='\t', header=0)
        self.M=dict_new['SMILES']
        self.T=dict_new['description']
        self.tokenizer = tokenizer
        self.len = max_length
        self.tag = tag #1 for M2T, 0 for T2M
        self.few = few

    def __len__(self):
        return int(len(self.M)*self.few)

    def __getitem__(self, index):
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len)*(-100)
        if self.tag:
            inp = self.tokenizer.encode(self.M[index])
            out = self.tokenizer.encode(self.T[index])
        else:
            inp = self.tokenizer.encode(self.T[index])
            out = self.tokenizer.encode(self.M[index])   
        inp_ids[:min(self.len, len(inp))] = torch.tensor(inp[:min(self.len, len(inp))])
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.tensor(out[:min(self.len, len(out))])
        
        return inp_ids.long(), att_msk.long(), labels.long()

def do_eval(model, dataloader, tokenizer, pth, tag, iftest=False):
    #tag=1 for M2T
    fw = open(pth, 'w') 
    if tag:
        fw.write('SMILES' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
    else:
        fw.write('description' + '\t' + 'ground truth' + '\t' + 'output' + '\n')
    model.eval()
    if tag:
        mx = 256
    else:
        mx = 128
    hit3 = 0
    hit1 = 0
    hitcnt = 1
    with torch.no_grad():
        for idx, i in enumerate(tqdm(dataloader)):
            inputs, atts, labels = i
            lab = (labels>0).long()*labels
            defau = 1
            if iftest:
                defau = 3
            output = model.generate(input_ids=inputs.cuda(), attention_mask=atts.cuda(), max_length=mx, num_beams=defau, num_return_sequences=1)
            for jdx, out in enumerate(output):
                
                if tag:
                    fw.write(tokenizer.decode(inputs[jdx]).split('</s>')[0].strip('<pad> ') + '\t' + tokenizer.decode(lab[jdx]).split('</s>')[0].strip('<pad> ') + '\t' + tokenizer.decode(out).split('</s>')[0].strip('<pad> ') + '\n')
                else:
                    tmplab = tokenizer.decode(lab[jdx])
                    pred = tokenizer.decode(out)
                    
                    fw.write(tokenizer.decode(inputs[jdx]).split('</s>')[0].strip('<pad> ') + '\t' + tmplab.split('</s>')[0].strip('<pad> ').replace('<unk> ', '\\') + '\t' + pred.split('</s>')[0].strip('<pad> ').replace('<unk> ', '\\') + '\n')
    fw.close()
    if tag:
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                text_translation_metrics.evaluate(
                    tokenizer, pth, mx
                )
        print(bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score)
        return bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score
    else:
        bleu_score, exact_match_score, levenshtein_score = mol_translation_metrics.evaluate(pth)
        validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(pth, 2)
        #fcd_metric_score = fcd_metric.evaluate(pth)
        print(bleu_score, exact_match_score, levenshtein_score, validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score)#, fcd_metric_score)
        return bleu_score, exact_match_score, levenshtein_score, validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score#, fcd_metric_score


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print('tokenizer loading')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    print('model loading')
    if args.version=='base':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    elif args.version=='large':
        model = T5ForConditionalGeneration.from_pretrained('t5-large')
    else:
        if args.tag:
            model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base-smiles2caption')
            tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base-smiles2caption", model_max_length=512)
        else:
            model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base-caption2smiles')
            tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base-caption2smiles", model_max_length=512)

    print('ckpt loading')
    
    if args.init_checkpoint is not None:
        pt = torch.load(args.init_checkpoint).state_dict()#['state_dict'] 
        model.load_state_dict(pt, strict=True)
        pt = None
    

    model = model.cuda()
    params = []
    for n,p in model.named_parameters():
        params.append(p)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    global_step = 0

    print('dataset loading')
    TrainSet = convert_dataset(args.pth_train, tokenizer, args.tag, few=args.few)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, pin_memory=True)
    DevSet = convert_dataset(args.pth_dev, tokenizer, args.tag)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, pin_memory=True)
    TestSet = convert_dataset(args.pth_test, tokenizer, args.tag)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, pin_memory=True)
    
    avg_loss = 0
    tag = 0
    max_rec = 0
    early = 0
    
    print('start training')
    accu=16/args.batch_size
    for epoch in range(args.epoch):
        model.train()
        if tag:
            break
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            if tag:
                break
            inputs, atts, labels = i
            output = model(input_ids=inputs.cuda(), attention_mask=atts.cuda(), labels=labels.cuda(), return_dict=True)
            loss = output.loss.mean()/accu
            loss.backward()
            avg_loss += loss.item()
            if idx%accu==0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step%25==0:#25
                    print('Step: ', global_step, ', loss: ', avg_loss)
                    avg_loss = 0

                if global_step>args.global_step:
                    return
                # do_eval
                if global_step % int(args.eval_len*args.few) == 0:#700 1750
                    if global_step>args.step_pre:
                        evs = do_eval(model, dev_dataloader, tokenizer, args.pth_out, args.tag)
                        model.train()
                        print('Step:', global_step, evs, early)
                        if evs[0]>max_rec:
                            early = 0
                            max_rec = evs[0]
                            torch.save(model, args.save_pth)
                        else:
                            early+=1
                            if early==5:
                                tag=1
    evs = do_eval(model, dev_dataloader, tokenizer, args.pth_out, args.tag)
    print('Step:', global_step, evs)
    if evs[0]>max_rec:
        max_rec = evs[0]
        torch.save(model, args.save_pth)
    
    model.load_state_dict(torch.load(args.save_pth).state_dict())
    evs = do_eval(model, test_dataloader, tokenizer, args.pth_out, args.tag, iftest=True)
    print('Test: ', evs)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default='save_model/preM2T_base.pt', type=str,)
    parser.add_argument("--version", default='base', type=str,)#base, large, molT5
    parser.add_argument("--save_pth", default='save_model/mol_M2T_pre_base.pt', type=str,)
    parser.add_argument("--lr", default=5e-4, type=float,)
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--tag", default=1, type=int,)
    parser.add_argument("--pth_train", default='data/train.txt', type=str,)
    parser.add_argument("--pth_dev", default='data/validation.txt', type=str,)
    parser.add_argument("--pth_test", default='data/test.txt', type=str,)
    parser.add_argument("--pth_out", default='log/mol_pre_base_M2T.txt', type=str,)
    parser.add_argument("--eval_len", default=1750, type=int,)
    parser.add_argument("--batch_size", default=16, type=int,)
    parser.add_argument("--epoch", default=50, type=int,)
    parser.add_argument("--seed", default=1111, type=int,)
    parser.add_argument("--global_step", default=200001, type=int,)
    parser.add_argument("--step_pre", default=5000, type=int,)#5000
    parser.add_argument("--few", default=1, type=float,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
