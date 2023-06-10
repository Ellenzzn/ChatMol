from transformers import T5Tokenizer
import joblib
import random
import numpy as np
tokenizer = T5Tokenizer.from_pretrained('t5-base')
jb = [[],[],[],[],[], []]
lines = open('more_demo.txt').readlines()
'''
lines += open('more_med12.txt').readlines()
lines += open('more_16.txt').readlines()
'''

for i in range(len(lines)):
    if i%20!=0:
        continue
    line = lines[i].strip('\n')
    if len(line.split(' '))<20:
        continue
    tok = tokenizer.encode(line)[:510]
    inp = []
    out = []
    pos = 0
    mark = 32099
    while pos<len(tok)-3:
        rd = random.randint(1,3)
        ms = random.randint(0,99)
        if ms<15 and mark>32069:
            inp += [mark]
            out += [mark]
            for tmp in range(rd):
                out += [tok[pos+tmp]]
            mark-=1
        else:
            for tmp in range(rd):
                inp += [tok[pos+tmp]]
        pos += rd
    jb[0].append(inp)
    jb[1].append(out)

print(len(jb[0]))

import pickle

dic = pickle.load(open('SM_result.pkl', 'rb'))
i =0

for ky in dic.keys():
    i+=1
    if i%3!=2:
        continue
    smi = dic[ky]['SMILES']
    tok = []
    tok = tokenizer.encode(smi)[:512]
    inp = []
    out = []
    pos = 0
    mark = 32099
    while pos<len(tok)-2:
        rd = random.randint(1,2)
        ms = random.randint(0,99)
        if ms<15 and mark>32069:
            inp += [mark]
            out += [mark]
            for tmp in range(rd):
                out += [tok[pos+tmp]]
            mark-=1
        else:
            for tmp in range(rd):
                inp += [tok[pos+tmp]]
        pos += rd
    jb[0].append(inp)
    jb[1].append(out)

print(len(jb[0]))
joblib.dump(jb, 'mixmsk.jbl')
