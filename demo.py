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
import pandas as pd


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print('tokenizer loading')
    M2Ttokenizer = T5Tokenizer.from_pretrained('t5-base')
    T2Mtokenizer = T5Tokenizer.from_pretrained('t5-base')

    print('model loading')
    if args.version=='base':
        M2Tmodel = T5ForConditionalGeneration.from_pretrained('t5-base')
        T2Mmodel = T5ForConditionalGeneration.from_pretrained('t5-base')
    else:
        M2Tmodel = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base-smiles2caption')
        M2Ttokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base-smiles2caption", model_max_length=512)
        T2Mmodel = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-base-caption2smiles')
        T2Mtokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base-caption2smiles", model_max_length=512)

    print('ckpt loading')
    if args.M2Tinit_checkpoint is not None:
        pt = torch.load(args.M2Tinit_checkpoint).state_dict()#['state_dict'] 
        M2Tmodel.load_state_dict(pt, strict=True)
        pt = None
    if args.T2Minit_checkpoint is not None:
        pt = torch.load(args.T2Minit_checkpoint).state_dict()#['state_dict']
        T2Mmodel.load_state_dict(pt, strict=True)
        pt = None


    M2Tmodel = M2Tmodel.cuda()
    T2Mmodel = T2Mmodel.cuda()
    history = ''
    last_mol = ''
    print("Input 'describe' to trigger molecule understanding of the current molecule, otherwise to use molecule generation. Input 'END' to clear dialog memory.")
    while True:
        txt = input("user: ")
        if txt.lower()=='end':
            history = ''
            last_mol = ''
            print('\n')
            continue
        if txt.find('describe')>-1 or txt.find('description')>-1:
            tok = M2Ttokenizer.encode(last_mol)
            inp = torch.tensor(tok).unsqueeze(0).long()[-512:]
            out = M2Tmodel.generate(input_ids=inp.cuda(), max_length=256, num_beams=3, num_return_sequences=1)
            pred = M2Ttokenizer.decode(out[0]).split('</s>')[0].strip('<pad> ')
            print('system: '+pred)
        else:
            tmp = history+' '+txt
            if len(last_mol)>0:
                tmp+=' It looks like '+last_mol+'. '
            tok = T2Mtokenizer.encode(tmp)
            inp = torch.tensor(tok).unsqueeze(0).long()[-512:]
            out = T2Mmodel.generate(input_ids=inp.cuda(), max_length=128, num_beams=3, num_return_sequences=3)
            mol1 = T2Mtokenizer.decode(out[0]).split('</s>')[0].strip('<pad> ').replace('<unk> ', '\\')
            mol2 = T2Mtokenizer.decode(out[1]).split('</s>')[0].strip('<pad> ').replace('<unk> ', '\\')
            mol3 = T2Mtokenizer.decode(out[2]).split('</s>')[0].strip('<pad> ').replace('<unk> ', '\\')
            history += ' '+txt
            print('system: choose one molecule to continue. 1. '+mol1+' 2. '+mol2+' 3. '+mol3)
            num = input('user: ')
            if num=='3':
                last_mol = mol3
            elif num=='2':
                last_mol = mol2
            else:
                last_mol = mol1
            

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--M2Tinit_checkpoint", default='save_model/understanding_base.pt', type=str,)
    parser.add_argument("--T2Minit_checkpoint", default='save_model/generation_noplug_base.pt', type=str,)
    parser.add_argument("--version", default='base', type=str,)#base, molt5
    parser.add_argument("--seed", default=1111, type=int,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
