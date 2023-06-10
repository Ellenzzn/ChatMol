f1=open("pred_dia_mol.txt","r")
f2=open("ans_dia_mol.txt","r")
f3=open("test_dia_out.txt","r")
fw=open("example_baseline_base.txt","w")
lines_ans=f2.readlines()
line_pred=f1.readlines()
lines_count=f3.readlines()
print("ground truth"+'\t'+"output",file=fw)
t=-1
for i in range (len(lines_ans)):
    len(lines_count[i].split('\t'))-1
    print(t)
    t+=len(lines_count[i].split('\t'))-1
    #print(len(line_pred))
    #print(len(lines_ans))
    print(lines_ans[t].strip('\n').replace("<unk> ","\\\\")+'\t'+line_pred[5*t].strip('\n').replace("<unk> ","\\\\"),file=fw)


