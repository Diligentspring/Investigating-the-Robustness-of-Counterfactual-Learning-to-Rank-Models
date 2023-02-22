DATA_PATH = sys.argv[1]
feature_lenth = int(sys.argv[2])
OUTPUT_PATH = sys.argv[3]

feature_fin = open(DATA_PATH, 'r')
qid_to_idx = {}
dids = []
initial_list = []
qids = []
labels = []
features = {}

line_num = -1
for line in feature_fin:
    line_num += 1
    arr = line.strip().split(' ')
    qid = arr[1].split(':')[1]
    if qid not in qid_to_idx:
        qid_to_idx[qid] = len(qid_to_idx)
        qids.append(qid)
        initial_list.append([])
        labels.append([])

    # create query-document information
    qidx = qid_to_idx[qid]

    # ignore this line if the number of documents reach rank_cut.
    # if rank_cut and len(self.initial_list[qidx]) >= rank_cut:
    #     continue

    label = int(arr[0])
    labels[qidx].append(label)
    did = qid + '_' + str(line_num)
    dids.append(did)
    initial_list[qidx].append(did)

    # read query-document feature vectors
    features[did] = [0.0 for _ in range(feature_lenth)]
    for x in arr[2:]:
        arr2 = x.split(':')
        feautre_idx = int(arr2[0]) - 1
        features[did][int(feautre_idx)] = float(arr2[1])

    if line_num % 10000 == 0:
        print('Reading finish: %d lines' % line_num)
feature_fin.close()

rank_list_size = -1
initial_list_lengths = [
    len(initial_list[i]) for i in range(len(initial_list))]
for i in range(len(initial_list_lengths)):
    x = initial_list_lengths[i]
    if rank_list_size < x:
        rank_list_size = x
print('rank_list_size: ' + str(rank_list_size))
print('Data reading finish!')

feature_file = open(OUTPUT_PATH+'.feature', 'w')
for d in features.keys():
    feature_file.write(d+' ')
    f = features[d]
    for i in range(feature_lenth):
        feature_file.write(str(i)+':')
        feature_file.write(str(f[i]) + ' ')
    feature_file.write('\n')
feature_file.close()

init_file = open(OUTPUT_PATH+'.init_list', 'w')
for qid in qid_to_idx.keys():
    init_file.write(qid+':')
    qidx = qid_to_idx[qid]
    dids = initial_list[qidx]
    for d in dids:
        init_file.write(d + ' ')
    init_file.write('\n')
init_file.close()

label_file = open(OUTPUT_PATH+'.labels', 'w')
for qid in qid_to_idx.keys():
    label_file.write(qid+':')
    qidx = qid_to_idx[qid]
    ls = labels[qidx]
    for l in ls:
        label_file.write(str(l) + ' ')
    label_file.write('\n')
label_file.close()