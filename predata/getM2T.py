from transformers import T5Tokenizer
import joblib
import random

tokenizer = T5Tokenizer.from_pretrained('t5-base')
jb = [[],[]]

import pickle

cleandic = pickle.load(open('clean_id2sm.pkl', 'rb'))
dic = pickle.load(open('propdic.pkl', 'rb'))
kys = list(dic.keys())
i = 0
tmp = []
for ky in dic.keys():
    if i%1000==0:
        print(i)
    i+=1
    if 'SMILES' in dic[ky]:
        smi = dic[ky]['SMILES']
    elif ky in cleandic:
        smi = cleandic[ky]
    else:
        print(ky)
        continue
    tag = False
    for topic in dic[ky].keys():
        if topic=='SMILES' or topic=='name':
            continue
        if topic not in tmp:
            tmp.append(topic)
        
        inp = [32067]+tokenizer.encode(topic+': '+smi)[:510]
        out = tokenizer.encode(dic[ky][topic])
        jb[0].append(inp)
        jb[1].append(out)

print(len(jb[0]))

dic = pickle.load(open('SM_result.pkl', 'rb'))
namedic = pickle.load(open('allcid2name.pkl', 'rb'))
kys = list(dic.keys())
i = 0
for ky in dic.keys():
    i+=1
    if i%3==0:
        #print(i)
        smi = dic[ky]['SMILES']
        if ky not in namedic:
            continue
        name = namedic[ky]#dic[ky]['name']
        inp = [32067]+tokenizer.encode('name: '+smi)[:510]
        out = tokenizer.encode(name)
        jb[0].append(inp)
        jb[1].append(out)
print(len(jb[0]))
joblib.dump(jb, 'preM2T.jbl')
