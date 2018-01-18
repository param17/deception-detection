#CSCI 5832 Natural Language Processing
#@author Paramjot Singh

#Hotel review Deception detection

import math
import re

def count(train):
    word_dict = {}
    train_file = open(train, 'r')

    for line in train_file:
        #removing all the punctuations to avoid difference between wife and wife,
        #it wont help in sentiment analysis, but will reduce the the size of vocabulary
        line = re.sub('[,.:;?!]','',line)
        #changing words to lower case and splitting based on spaces
        line = line.lower().strip().split()
        line = line[1:]  #to avoid ID value for line

        for word in line:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1

    train_file.close()

    return word_dict


def read(train):

    word_dict = {}
    train_file = open(train, 'r')

    for line in train_file:
        #removing all the punctuations to avoid difference between wife and wife,
        #it wont help in sentiment analysis, but will reduce the the size of vocabulary
        line = re.sub('[,.:;?!]','',line)
        #changing words to lower case and splitting based on spaces
        line = line.lower().strip().split()
        id=line[0].upper()
        line = line[1:]  #to avoid ID value for line

        for word in line:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        # print(id)
        if id in pos_f_bucket:
            pos_f_train.update(word_dict)
        elif id in pos_t_bucket:
            pos_t_train.update(word_dict)
        elif id in neg_f_bucket:
            neg_f_train.update(word_dict)
        elif id in neg_t_bucket:
            neg_t_train.update(word_dict)
    train_file.close()

    return neg_f_train


def prob_calc(train, test_data, train_vocab):

    prob_list = []
    word_count = 0
    for word in train:
        word_count += train[word]

    for word in test_data:
        word_freq = 0
        if word in train:
            word_freq = train[word]
        else:
            word_freq = 0

        prob = (word_freq + 1)/(word_count+len(train_vocab))
        prob_list.append(prob)
    return prob_list


def print_result(result):
    with open('singh-paramjot-extra-out.txt', 'w') as file:
        for id in result:
            file.write(id + '\t' + result[id] +'\n')
    file.close()


def sentiment_calc(test_data):
    result = {}
    for line in test_data:
        pos_prob_list = prob_calc(pos_train, test_data[line], train_vocabulary)
        neg_prob_list = prob_calc(neg_train, test_data[line], train_vocabulary)

        #no need to add prior in this case since its same lg(0.5)
        pos_prob = 0
        neg_prob = 0
        for prob in pos_prob_list:
            pos_prob += math.log(prob)

        for prob in neg_prob_list:
            neg_prob += math.log(prob)

        # print(line)
        if pos_prob > neg_prob:
            # print(line.upper()+'\tPOS\n')
            result[line.upper()] = 'POS'

        elif neg_prob > pos_prob:
            # print(line.upper() + '\tNEG\n')
            result[line.upper()] = 'NEG'

    return result

def deception_calc(test_data):
    result = {}
    for line in test_data:
        pos_f_prob_list = prob_calc(pos_f_train, test_data[line], train_vocabulary)
        pos_t_prob_list = prob_calc(pos_t_train, test_data[line], train_vocabulary)
        neg_f_prob_list = prob_calc(neg_f_train, test_data[line], train_vocabulary)
        neg_t_prob_list = prob_calc(neg_t_train, test_data[line], train_vocabulary)

        #no need to add prior in this case since its same lg(0.5)
        true_prob = 0
        false_prob = 0
        for prob in neg_t_prob_list:
            true_prob += math.log(prob)
        for prob in pos_t_prob_list:
            true_prob += math.log(prob)

        for prob in pos_f_prob_list:
            false_prob += math.log(prob)
        for prob in neg_f_prob_list:
            false_prob += math.log(prob)

        # print(line)
        if true_prob > false_prob:
            # print(line.upper()+'\tPOS\n')
            result[line.upper()] = 'T'

        elif false_prob > true_prob:
            # print(line.upper() + '\tNEG\n')
            result[line.upper()] = 'F'

    return result


def classify(result, tf_class):

    for item in result:
        if tf_class == 'f':
            if result[item] == 'POS':
                pos_f_bucket.append(item)
            elif result[item] == 'NEG':
                neg_f_bucket.append(item)
        elif tf_class == 't':
            if result[item] == 'POS':
                pos_t_bucket.append(item)
            elif result[item] == 'NEG':
                neg_t_bucket.append(item)


if __name__ == "__main__":

    pos_f_bucket = []
    pos_t_bucket = []
    neg_f_bucket = []
    neg_t_bucket = []

    ############### TRAIN ASSGN-3 TEST DATA for POS or NEG Review #################

    #count positive train data words
    pos_train = count('hotelPos-train.txt')
    #count negative train data words
    neg_train = count('hotelNeg-train.txt')

    ps = [(k, pos_train[k]) for k in sorted(pos_train, key=pos_train.get, reverse=True)]

    ns = [(k, neg_train[k]) for k in sorted(neg_train, key=neg_train.get, reverse=True)]

    #join both dictionary to form vocabulary
    train_vocabulary = pos_train.copy()
    train_vocabulary.update(neg_train)

    s = [(k, train_vocabulary[k]) for k in sorted(train_vocabulary, key=train_vocabulary.get, reverse=True)]
    # print(s)

    ############### TEST with TRAIN-DATA for Assign 5 ######################

    #get test data
    test_data = {}
    test_file = open('hotelF-train.txt', 'r')
    for line in test_file:
        line = re.sub('[.,;:!?]', '', line)
        line = line.lower().strip().split()

        test_data[line[0]] = line[1:]
    test_file.close()

    result = sentiment_calc(test_data)

    classify(result,'f')

    #get test data
    test_data = {}
    test_file = open('hotelT-train.txt', 'r')
    for line in test_file:
        line = re.sub('[.,;:!?]', '', line)
        line = line.lower().strip().split()

        test_data[line[0]] = line[1:]
    test_file.close()

    result = sentiment_calc(test_data)

    classify(result, 't')

    ################ TRAIN with Assign 5 train data using 4 buckets ################

    pos_f_train = {}
    pos_t_train = {}
    neg_f_train = {}
    neg_t_train = {}

    read('hotelT-train.txt')
    read('hotelF-train.txt')

    pfs = [(k, pos_f_train[k]) for k in sorted(pos_f_train, key=pos_f_train.get, reverse=True)]
    pts = [(k, pos_t_train[k]) for k in sorted(pos_t_train, key=pos_t_train.get, reverse=True)]
    nfs = [(k, neg_f_train[k]) for k in sorted(neg_f_train, key=neg_f_train.get, reverse=True)]
    nts = [(k, neg_t_train[k]) for k in sorted(neg_t_train, key=neg_t_train.get, reverse=True)]

    train_vocabulary = pos_f_train.copy()
    train_vocabulary.update(pos_t_train)
    train_vocabulary.update(neg_f_train)
    train_vocabulary.update(neg_t_train)

    s = [(k, train_vocabulary[k]) for k in sorted(train_vocabulary, key=train_vocabulary.get, reverse=True)]
    # print(s)

    # final test data
    final_test_data = {}
    test_file = open('hotelDeceptionTest.txt', 'r')
    for line in test_file:
        line = re.sub('[.,;:!?]', '', line)
        line = line.lower().strip().split()
        final_test_data[line[0]] = line[1:]
    test_file.close()

    final_result = deception_calc(final_test_data)

    print_result(final_result)
