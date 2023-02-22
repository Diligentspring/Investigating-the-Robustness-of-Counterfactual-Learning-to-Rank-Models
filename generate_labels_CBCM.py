from click_simulator import ComparisionBasedClickModel
import sys
import random

random.seed(1)

initlist_path = sys.argv[1]
w= float(sys.argv[2])
# beta = float(sys.argv[3])
g = float(sys.argv[3])
eta = float(sys.argv[4])

train_init_file = open("train.init_list", "r", encoding="utf-8")
label_file = open("train.labels", "r", encoding="utf-8")
qid_dic = {}
label_dic = {}
for line in train_init_file:
    if(line != ''):
        qid = line.strip().split(":")[0]
        docs = line.strip().split(":")[1].strip().split(" ")
        qid_dic[qid] = int(docs[0].split("_")[1])
        for d in docs:
            label_dic[d] = -1
train_init_file.close()

for line in label_file:
    if(line != ''):
        qid = line.strip().split(":")[0]
        labels = line.strip().split(":")[1].strip().split(" ")
        for i in range(len(labels)):
            label_dic[qid+'_'+str(qid_dic[qid]+i)] = int(labels[i])
label_file.close()

CBCM = ComparisionBasedClickModel()

CBCM.setExamProb(eta)
# CBCM.setG(beta)
CBCM.setG(g)
CBCM.setSatProb(w)

initlist_file = open(initlist_path, 'r')

if w == 0.2:
    gen_label_path = 'CBCM_w0d2_eta0d5.labels'
elif w == 0.4:
    gen_label_path = 'CBCM_w0d4_eta0d75.labels'
else:
    gen_label_path = 'CBCM_w0d6_eta1.labels'

gen_label_file = open(gen_label_path, 'w')

# gen_label_file = open('CBCM_w0d2_eta0d5.labels', 'w')
# gen_label_file = open('CBCM_w0d4_eta0d75.labels', 'w')
# gen_label_file = open('CBCM_w0d6_g6d6_eta1.labels', 'w')

for line in initlist_file:
    if line != '':
        query = line.strip().split(':')[0]
        gen_label_file.write(query+':')
        docs = line.strip().split(':')[1].strip().split(' ')
        labels = [label_dic[doc] for doc in docs]
        click_list = CBCM.sampleClicksForOneList(labels)
        for c in click_list:
            gen_label_file.write(str(int(c))+' ')
        gen_label_file.write('\n')

initlist_file.close()
gen_label_file.close()












