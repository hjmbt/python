from exercise_numpy import *


def load_dataset():
    postinglist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupi']]
    classvec = [0, 1, 0, 1, 0, 1]
    return postinglist, classvec


def create_vocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


def setof_words2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!") % word
    return returnvec


listoposts, listclasses = load_dataset()
myvocablist = create_vocablist(listoposts)


# print(myvocablist)
# print(setof_words2vec(myvocablist, listoposts[0]))


def trainNB0(trainmatrix, traincategory):
    num_train_docs = len(trainmatrix)
    num_words = len(trainmatrix)
    pabusive = sum(traincategory) / float(num_train_docs)
    p0num = zeros(num_words)
    p1num = zeros(num_words)
    p0denom = 0.0
    p1denom = 0.0
    for i in range(num_train_docs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = p1num / p1denom
    p0vect = p0num / p0denom
    return p0vect, p1vect, pabusive
