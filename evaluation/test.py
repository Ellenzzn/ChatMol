"""

"""

import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os

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

def getitem(text,tokenizer):
        inp = tokenizer.encode(text.strip('\n'))
        if len(inp)>512:
            inp=inp[-512:]
        inp_ids = torch.ones(512)
        att_msk = torch.zeros(512)  
        inp_ids[:min(512, len(inp))] = torch.from_numpy(np.array(inp))[:min(512, len(inp))]
        att_msk[:min(512, len(inp))] = 1
        

        return torch.unsqueeze(inp_ids.long(),0), torch.unsqueeze(att_msk.long(),0)

f=open("result_"+args.pth+".txt","w")
def do_eval(tokenizer, model):
    cnt = [0,0,0,0,0,0,0,0,0]
    bleu2 = [0,0,0,0,0,0,0,0,0]
    bleu4 = [0,0,0,0,0,0,0,0,0,0]
    nist = [0,0,0,0,0,0,0,0,0,0,0]
    acc = [0,0,0,0,0,0,0,0,0,0,0,0]
    acc3=[0,0,0,0,0,0,0,0,0,0,0,0]
    acc5=[0,0,0,0,0,0,0,0,0,0,0,0]
    fin=open("../data/test_inp.txt","r")
    fout=open("../data/test_out.txt","r")
    f1 = open('inp_'+args.pth+'.txt', 'w')
    f2 = open('pred_'+args.pth+'.txt', 'w')
    f3 = open('ans_'+args.pth+'.txt', 'w')
    rec = []
    line_ins=fin.readlines()
    line_outs=fout.readlines()
    with torch.no_grad():
        
        for id in range(len(line_ins)):  
            
         pre=""
         print("dialo",id,":",end="",file=f)
         inp_seri=line_ins[id].split('\t')
         out_seri=line_outs[id].split('\t')
         for idx in range(len(inp_seri)-1):
            inputs, atts = getitem(inp_seri[idx]+pre,tokenizer)
            labels,att=getitem(out_seri[idx],tokenizer)
            output = model.generate(input_ids=inputs.cuda(), attention_mask=atts.cuda(), max_length=512,
                                           num_beams=5, num_return_sequences=5)
            tmp = (labels > 0) * labels + (labels < 0)
            inp = [tokenizer.decode(piece).split('</s>')[0] for piece in inputs]
            pred = [tokenizer.decode(piece).split('</s>')[0].strip('<pad> ') for piece in output]
            ans = [tokenizer.decode(piece).split('</s>')[0] for piece in tmp]
            pre="It looks like "+pred[0]+"."
            for st in inp:
                f1.write(st + '\n')
                f1.flush()     
            for st in pred:
                f2.write(st + '\n')
                f2.flush()
            for st in ans:
                f3.write(st + '\n')
                f3.flush()
            
            if idx ==len(inp_seri)-2:
             for one in range(len(ans)):
                if pred[5 * one] == ans[one]:
                    acc[idx] += 1
                    
                    print("correct"+'\t',end="",file=f)
                else:
                    print("wrong"+'\t',end="",file=f)
                if ans[one] in pred[5 * one:5 * one+3]:
                    acc3[idx]+=1
                if ans[one] in pred[5 * one:5 * one+5]:
                    acc5[idx]+=1
                if sum(output[one] > 1) < 4:
                    cnt[idx] += 1
                    continue

                pred1 = [pred[5 * one].strip('\n')]
                gt1 = [[ans[one].strip('\n')]]
                try:
                    bleu2[idx] += nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 1, 0, 0))
                    bleu22=nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 1, 0, 0))
                    print("bleu2:",bleu22,'\t',end="",file=f)
                except ZeroDivisionError:
                    0
                try:
                    bleu4[idx] += nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 0, 0, 1))
                    bleu44=nltk.translate.bleu_score.corpus_bleu(gt1, pred1,
                                                                   smoothing_function=SmoothingFunction().method1,
                                                                   weights=(0, 0, 0, 1))
                    print("bleu4:",bleu44,'\t',end="",file=f)
                except ZeroDivisionError:
                    0
                try:
                    nist[idx] += nltk.translate.nist_score.corpus_nist(gt1, pred1, n=4)
                    nis=nltk.translate.nist_score.corpus_nist(gt1, pred1, n=4)
                    print("nist:",nis,"\t",file=f)

                except ZeroDivisionError:
                    0
                rec.append([wd for wd in pred1[0]])
                cnt[idx] += 1
        print("",file=f)
        for i in range(8):
         if cnt[i]>0:
          print("turn",i,acc[i] / cnt[i],acc3[i]/cnt[i],acc5[i]/cnt[i], bleu2[i] / cnt[i], bleu4[i] / cnt[i], nist[i] / cnt[i],file=f)
  


def main(args):
    
    device = torch.device('cuda')

    tokenizer =  T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
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
        pt = torch.load(args.init_checkpoint).state_dict()
        pt1 = {k[7:]: v for k, v in pt.items()}
        model.load_state_dict(pt1, strict=True)
    pt = None
    pt1 = None

    model = model.cuda()
    global_step = 0
    do_eval(tokenizer, model)
    
    


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--pth", default='prebase', type=str, )
    parser.add_argument("--init_checkpoint", default='../save_model/dia_pre_base.pt', type=int,)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
