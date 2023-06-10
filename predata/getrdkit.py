import pickle
from transformers import T5Tokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import random
import joblib
jb = [[],[]]

dic = pickle.load(open('SM_result.pkl', 'rb'))

tokenizer = T5Tokenizer.from_pretrained('t5-base')
prefix = [32068]+tokenizer.encode('spatial information: ')
smcnt = 0

i = 0
for ky in dic.keys():
    i+=1
    if i%3!=1:
        continue
    smi = dic[ky]['SMILES']
    try:
        rdkit_mol =  Chem.MolFromSmiles(smi)
    except Exception as e:
        continue
    if rdkit_mol is None:
        continue
    #rdkit_mol = Chem.AddHs(rdkit_mol)
    #rdkit_mol1 = Chem.RemoveHs(rdkit_mol)
    #AllChem.EmbedMolecule(rdkit_mol)
    atoms = rdkit_mol.GetAtoms()
    if len(atoms)<3:
        continue
    pos = -1
    posrec = []
    atmrec = []
    for di in range(len(atoms)):
        at = atoms[di].GetSymbol()
        pos = smi.find(at,pos+1)
        posrec.append(pos)
        atmrec.append(at)
    seq = np.arange(len(atoms))
    np.random.shuffle(seq)
    indrec = []
    outrec = []
    for di in range(max(2, int(len(posrec)*0.3))):
        ind = int(seq[di])
        indrec.append(ind)
        tmpstr = atmrec[ind]+':'
        for nei in atoms[ind].GetNeighbors():
            tmpstr+=nei.GetSymbol()
        tmpstr+=' aromatic '
        if atoms[ind].GetIsAromatic():
            tmpstr+='true ring '
        else:
            tmpstr+='false ring '
        ring = 0
        if atoms[ind].IsInRing():
            for ri in range(3,9):
                if atoms[ind].IsInRingSize(ri):
                    ring = ri
        tmpstr+=str(ring)
        outrec.append(tmpstr)
    indrec = np.array(indrec)
    arg = np.argsort(indrec)
    pos = 0
    inp = ''
    out = ''
    for ag in arg:
        inp+=smi[pos:posrec[indrec[ag]]]
        inp+='<'
        inp+=atmrec[indrec[ag]]
        inp+='>'
        pos = posrec[indrec[ag]]+len(atmrec[indrec[ag]])
        out+=outrec[ag]+'|'
    inp+=smi[pos:]
    tok = prefix+tokenizer.encode(inp)
    lab = tokenizer.encode(out)
    jb[0].append(tok)
    jb[1].append(lab)
    smcnt+=len(lab)
print(len(jb[0]), smcnt/len(jb[0]))
joblib.dump(jb, 'spat.jbl')
