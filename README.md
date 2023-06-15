# ChatMol

Code and data for [Interactive Molecular Discovery with Natural Language](TODO). 

## Requirement
Main packages:
python==3.9.0;
torch==1.13.0;
transformers==4.28.0;
pandas==1.5.2;
numpy==1.23.5;
nltk==3.7

## Data Preprocess
### Pre-training
Related codes and data instances are in `predata/`. 
- `getmsk.py`: Masked language modeling.
- `getM2T.py`: Experimental property prediction & SMILES-to-name.
- `getT2M.py`: Literature-to-SMILES mapping correlation.
- `getrdkit.py`: Spatial structure prediction.
### Fine-tuning
Related codes and data instances are in `data/`.
- train/validation/test.txt are ChEBI-20k dataset.
- PCtrain/PCdev/PCtest.txt are PCdes dataset.
- Files under `ChEBI-dia/` are the new ChEBI-dia dataset. 

## Model Training and Inferencing
We stronly recommend using GPUs for this stage. First create an empty folder `save_model/`.

For knowledgeable and versatile pre-training, the jbl data files have to be first put under `predata/`. Change the `pth` args to combine different pre-training tasks. For instance:
```
CUDA_VISIBLE_DEVICES=0 python multi_pretraining.py --version base --save_pth save_model/understanding_pre.pt --pth predata/mixmsk.jbl --pth2 predata/preM2T.jbl --pth3 predata/spat.jbl
```

For molecule understanding tuning, remember to adjust the evaluate step length for different downstream datasets. For instance:
```
CUDA_VISIBLE_DEVICES=0 python molecule_understanding --pth_train data/PCtrain.txt --pth_dev data/PCdev.txt --pth_test data/PCtest.py --eval_len 700 --init_checkpoint save_model/pre_understanding_base.pt --save_pth save_model/understanding_
```

For molecule generation tuning, first conduct the fine-tuning, and then separately generate the test result and do evaluation for the specific checkpoint. For instance:
```
CUDA_VISIBLE_DEVICES=0 python molecule_generation.py --save_pth save_model/generation_noplug_ 
cd evaluation/
python test.py --init_checkpoint ../save_model/generation_noplug_100000.pt
```

You can also download the reference checkpoint from our [Google Drive](https://drive.google.com/drive/folders/1I-LcBE0emj8p1W6WFCbEYajTADGN1RuC?usp=sharing) and put them under `save_model/`. Run the `demo.py` and provide your input according to the hints:
```
Input 'describe' to trigger molecule understanding of the current molecule, otherwise to use molecule generation. Input 'END' to clear dialog memory.
user: It derives from a tryptamine.
system: choose one molecule to continue. 1. C1=CC=C2C(=C1)C(=CN2)CCN 2. CNCCC1=CNC2=CC=CC=C21 3. C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N
user: 2
user: It is an organic phosphate, a tryptamine alkaloid and a secondary amino compound.
system: choose one molecule to continue. 1. C1=CC=C2C(=C1)C(=CN2)CCN 2. C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N 3. CNCCC1=CNC2=CC=CC=C21
user: 3
user: describe the current molecule
system: The molecule is a tryptamine alkaloid and a member of tryptamines. It has a role as a metabolite. It derives from a tryptamine.
user: It has a role as a hallucinogen and a fungal metabolite.
system: choose one molecule to continue. 1. C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N 2. C[C@H](CC1=CNC2=CC=CC=C21)N[C@@H](C)C(=O)N[C@@H](CC3=CC=CC=C3)C(=O)O 3. C[C@H](CC1=CNC2=CC=CC=C21)OP(=O)(O)O
user: 3
user: END

user:  It derives from a formic acid.
system: choose one molecule to continue. 1. CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCSC=O)O 2. C(=O)[O-] 3. CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)N)O
user: 2
user: It has a role as a buffer.
system: choose one molecule to continue. 1. C(=O)[O-] 2. C(C(=O)O)N(C(=O)N)CC(=O)O 3. C(C(=O)O)N(CC(=O)O)CC(=O)O
user: 1
user: The molecule is the potassium salt of formic acid.
system: choose one molecule to continue. 1. C(=O)[O-].[K+] 2. [C@@H]([C@H]([C@@H]([C@H](C(=O)[O-])O)O)O)([C@H](C(=O)[O-])O)O.[K+] 3. [C@@H]([C@H]([C@@H]([C@H](C(=O)[O-])O)O)O)O.[K+]
user: 1
user: describe it
system: The molecule is a potassium salt comprising equal numbers of potassium and formic acid ions. It has a role as a fertilizer, a food acidity regulator, a human metabolite and a Saccharomyces cerevisiae metabolite. It contains a formate.
user: END
```

