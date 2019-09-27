from math import log
import operator


def calc_schannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for featVec in dataset:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = [u'no surfacing', u'filppers']
    return dataset, labels


mydat, labels = create_dataset()
print(mydat)
print(calc_schannon_ent(mydat))


# mydat[0][-1] = 'maybe'
# print(calc_schannon_ent(mydat))


# 按照给定特征划分数据集
def split_dataset(dataset, axis, value):
    ret_dataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reduced_featvec = featvec[:axis]
            reduced_featvec.extend(featvec[axis + 1:])
            ret_dataset.append(reduced_featvec)
    return ret_dataset


print(split_dataset(mydat, 0, 1))


# 选择最好的数据集划分方式
def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_schannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        featlist = [example[i] for example in dataset]
        unique_vals = set(featlist)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_schannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 如果最终分类依旧有多个分组 则以多数派为准
def majority_cnt(classlist):
    class_count = {}
    for vote in classlist:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建树的函数代码
def create_tree(dataset, labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majority_cnt(classlist)
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    mytree = {best_feat_label: {}}
    # del (labels[best_feat])
    featvals = [example[best_feat] for example in dataset]
    uniquevals = set(featvals)
    for value in uniquevals:
        sub_labels = labels[:]
        mytree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return mytree


mytree = create_tree(mydat, labels)


# 使用决策树的分类函数
def classify(inputTree, feat_labels, testvec):
    first_str = list(inputTree.keys())[0]
    second_dict = inputTree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if testvec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, testvec)
            else:
                class_label = second_dict[key]
    return class_label


# print(classify(mytree, labels, [1, 0]))
# ValueError: 'no surfacing' is not in list -> del (labels[best_feat])


# 使用pickle模块存储决策树
def store_tree(input_tree, filename):
    import pickle
    fw = open(filename)
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
