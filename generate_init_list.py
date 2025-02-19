import random
import math
import sys

random.seed(1)

ULTRA_DATA_PATH = sys.argv[1]
classify_path = sys.argv[2] # the file that stores the relevance scores given by the ranker (e.g. 1% DNN)
classify_file = open(classify_path, 'r')
session_num = int(sys.argv[3])
output_path = sys.argv[4]

train_initlist_file = open(ULTRA_DATA_PATH+"train.init_list", 'r')
pl_initlist_file = open(output_path + ".init_list", 'w')

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

        for x in range(session_num):
            sorted_id = sorted(range(len(labels_to_pl)), key=lambda x: labels_to_pl[x], reverse=True)
            new_init_list = [init_list[i] for i in sorted_id]
            pl_initlist_file.write(
                init_list[0].strip().split('_')[0] + '_' + str(x) + ':' + ' '.join(new_init_list) + '\n')
        index = index + 1

classify_file.close()
pl_initlist_file.close()