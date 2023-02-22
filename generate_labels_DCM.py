from click_simulator import DependentClickModel
import sys
import random
random.seed(1)

initlist_path = sys.argv[1]
beta = float(sys.argv[2])
eta = float(sys.argv[3])

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

DCM = DependentClickModel()

DCM.setExamProb(eta)
DCM.setContProb(beta, eta)

initlist_file = open(initlist_path, 'r')

# train_pl_path = initlist_path.strip().strip('cut_').strip('.init_list')
#
if beta == 0.6:
    gen_label_path1 = 'DCM_beta0d6_'
else:
    gen_label_path1 = 'DCM_beta{}_'.format(int(beta))
if eta == 0.5:
    gen_label_path2 = 'eta0d5.labels'
else:
    gen_label_path2 = 'eta{}.labels'.format(int(eta))

gen_label_file = open(gen_label_path1 + gen_label_path2, 'w')
# gen_label_file = open('t'+ train_pl_path + '/' + gen_label_path1 + gen_label_path2, 'w')
# gen_label_file = open('DCM_beta1_eta0d5.labels', 'w')
for line in initlist_file:
    if line != '':
        query = line.strip().split(':')[0]
        gen_label_file.write(query+':')
        docs = line.strip().split(':')[1].strip().split(' ')
        labels = [label_dic[doc] for doc in docs]
        click_list, exam_p_list, click_p_list = DCM.sampleClicksForOneList(labels)
        for c in click_list:
            gen_label_file.write(str(int(c))+' ')
        gen_label_file.write('\n')

initlist_file.close()
gen_label_file.close()












