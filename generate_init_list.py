import random
import math
import sys

random.seed(1)

ULTRA_DATA_PATH = sys.argv[1]
classify_path = sys.argv[2] # the file that stores the relevance scores given by the ranker (e.g. 1% DNN)
classify_file = open(classify_path, 'r')
w = sys.argv[3] # the weight parameter
output_path = sys.argv[4]

train_initlist_file = open(ULTRA_DATA_PATH+"train.init_list", 'r')
pl_initlist_file = open(output_path + ".init_list", 'w')

def plackett_luce(labels, weight):
    index_list = []
    index_remain = [_ for _ in range(len(labels))]

    exps = [math.exp(l*weight) for l in labels]
    s = sum(exps)
    for i in range(len(labels)):
        if s>0:
            probs = [e/s for e in exps]
        else:
            probs = [1/len(exps) for _ in exps]
        prob = random.random()
        for j in range(len(probs)):
            if prob <= sum(probs[0:(j+1)]):
                break
        index_list.append(index_remain[j])
        index_remain.pop(j)
        s = s - exps[j]
        exps.pop(j)
    return index_list

init_lists = []
for line in train_initlist_file:
    if line!= '':
        qid = line.strip().split(":")[0]
        init_list = line.strip().split(":")[1].strip().split(' ')
        init_lists.append(init_list)

train_initlist_file.close()

index = 0
for line in classify_file:
    if line != '':
        labels = line.strip().split(" ")
        labels_to_pl = []
        init_list = init_lists[index]
        for i in range(len(init_list)):
            labels_to_pl.append(float(labels[i]))

        for x in range(20):
            index_list = plackett_luce(labels_to_pl, float(w))
            new_init_list = [init_list[i] for i in index_list]
            pl_initlist_file.write(init_list[0].strip().split('_')[0] +'_'+str(x) + ':' + ' '.join(new_init_list) + '\n')

            #used for pl_inf
            # sorted_id = sorted(range(len(labels_to_pl)), key=lambda x: labels_to_pl[x], reverse=True)
            # new_init_list = [init_list[i] for i in sorted_id]
            # pl_initlist_file.write(
            #     init_list[0].strip().split('_')[0] + '_' + str(x) + ':' + ' '.join(new_init_list) + '\n')
        index = index + 1

classify_file.close()
pl_initlist_file.close()