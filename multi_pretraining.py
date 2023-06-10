import torch
import math
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

class pre_dataset(Dataset):
    def __init__(self, pth, pth2,pth3, max_length=512):
        self.data = joblib.load(pth)
        
        tmp = joblib.load(pth2)
        self.data[0]+=tmp[0]#[:96880]
        self.data[1]+=tmp[1]#[:96880]
        
        tmp = joblib.load(pth3)
        self.data[0]+=tmp[0]
        self.data[1]+=tmp[1]
        
        self.len = max_length
    
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len)*(-100)
        inp = self.data[0][index]
        out = self.data[1][index]
        inp_ids[:min(self.len, len(inp))] = torch.tensor(inp[:min(self.len, len(inp))])
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.tensor(out[:min(self.len, len(out))])
        
        return inp_ids.long(), att_msk.long(), labels.long()

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print('tokenizer loading')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    config = T5Config(
        vocab_size=32128,
        d_model=256,#768
        d_kv=16,#64
        d_ff=768,#3072
        num_layers=4,#12
        num_heads=4,#12
        relative_attention_num_buckets=8,#32
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=1,#32100,
        decoder_start_token_id=1,
        eos_token_id=2,
        gradient_checkpointing=False)
    
    print('model loading')
    if args.version=='base':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    elif args.version=='large':
        model = T5ForConditionalGeneration.from_pretrained('t5-large')
    else:
        model = T5ForConditionalGeneration(config)

    print('ckpt loading')
    
    if args.init_checkpoint is not None:
        pt = torch.load(args.init_checkpoint).state_dict()#['state_dict'] 
        #pt1 = {k[7:]: v for k, v in pt.items()}
        model.load_state_dict(pt, strict=True)
        pt = None

    model = model.cuda()
    params = []
    for n,p in model.named_parameters():
        params.append(p)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    #optimizer = Adafactor(model.parameters())#, scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
       
    global_step = 0

    print('dataset loading')
    TrainSet = pre_dataset(args.pth, args.pth2, args.pth3)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, pin_memory=True)
    avg_loss = 0
    tag = 0
    print('start training')
    accu=256/args.batch_size #256
    for epoch in range(args.epoch):
        if tag:
            break
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            if tag:
                break
            inputs, atts, labels = i
            output = model(input_ids=inputs.cuda(), attention_mask=atts.cuda(), labels=labels.cuda(), return_dict=True)
            loss = output.loss.mean()/accu
            if math.isnan(loss.item()):
                loss = None
                output = None
                continue
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
                if global_step % 300 == 0:#500
                    if global_step>args.step_pre:
                        torch.save(model, args.save_pth)
    

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--version", default='base', type=str,)#base, large, zero
    parser.add_argument("--save_pth", default='save_model/preM2T_base.pt', type=str,)
    parser.add_argument("--lr", default=5e-4, type=float,)#5e-4 2e-4
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--pth", default='predata/mixmsk.jbl', type=str,)
    parser.add_argument("--pth2", default='predata/preM2T.jbl', type=str,)
    parser.add_argument("--pth3", default='predata/spat.jbl', type=str,)
    parser.add_argument("--batch_size", default=16, type=int,)
    parser.add_argument("--epoch", default=1, type=int,)#20
    parser.add_argument("--seed", default=1234, type=int,)#1111
    parser.add_argument("--global_step", default=50001, type=int,)#1000
    parser.add_argument("--step_pre", default=0, type=int,)#300
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
