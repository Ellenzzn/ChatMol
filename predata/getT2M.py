from transformers import T5Tokenizer
import joblib
import random

tokenizer = T5Tokenizer.from_pretrained('t5-base')
jb = [[],[]]

lines = open('inp_0.txt').readlines()
gts = open('SM_0.txt').readlines()
'''
lines += open('inp_med12.txt').readlines()
gts += open('SM_med12.txt').readlines()
lines += open('inp_16.txt').readlines()
gts += open('SM_16.txt').readlines()
'''
print(len(lines))

for i in range(len(lines)):
    if i%1000==0:
        print(i)
    line = lines[i].strip('\n')
    inp = [32068]+tokenizer.encode('Chemical SMILES: '+line)[:510]
    out = tokenizer.encode(gts[i].strip('\n'))
    jb[0].append(inp)
    jb[1].append(out)

joblib.dump(jb, 'preT2M.jbl')
