import numpy as np
import random
import ijson
import tensorflow as tf
from tqdm import tqdm
import time
import pickle as pkl
import tokenization



tf.flags.DEFINE_string("train_file", "task-1.ubuntu.train.json", "path to train file")
tf.flags.DEFINE_string("valid_file", "task-1.ubuntu.dev.json", "path to valid file")


def process_dialog(dialog, data_type):
    """
    Add __eou__ and __eot__ tags between utterances and create context, correct response and wrong responses.
    """
    #  default is having right answer
    flag = 0

    # 1. Create the context (a list of utterances)
    utterances_id = dialog['example-id']
    context = dialog['messages-so-far']
    utterances = []
    response = []
    speaker = context[0]['speaker']
    for i in range(len(context)):
        if i == 0:
            continue

        if context[i]['speaker'] == speaker:
            continue

        if i < (len(context) - 1):

            if context[i]['speaker'] == context[i + 1]['speaker']:
                response.append(context[i]['utterance'].lower() + " __eou__")
            else:
                response.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        else:
            response.append(context[i]['utterance'].lower() + " __eou__ __eot__")


    for i in range(len(context)):


        if i < (len(context) - 1):

            if context[i]['speaker'] == context[i + 1]['speaker']:
                utterances.append(context[i]['utterance'].lower() + " __eou__")
            else:
                utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        else:
            utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")



    return utterances_id, response, utterances


def pasing_data(fname, data_type):


    if data_type == 'train':
        total_num = 220609
    elif data_type == 'valid':
        total_num = 4804
    HAE_dic = {}
    bad_sample = 0
    num_total = 0
    do_lower_case = True
    vocab_file = '../../../uncased_L-12_H-768_A-12/vocab.txt'
    with open(fname, 'rb') as f:
        json_data = ijson.items(f, 'item')
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        for i, entry in enumerate(json_data):
            if i % 3000 ==0:
                print('Done {}'.format(i))
                print('Bad_sample {}'.format(bad_sample))
            utterances_id, response, utterances = process_dialog(entry, data_type)

            sent = ' '
            context = sent.join(utterances)
            tokens = tokenizer.tokenize(context)
            REBED = [0]*len(tokens)
            for line in response:
                index = context.find(line)
                context1 = context[:index]
                context_len = tokenizer.tokenize(context1)
                response_len = tokenizer.tokenize(line)
                for i in range(len(response_len)):
                    REBED[i+len(context_len)]=1


            # if utterances_id in dic:
            HAE_dic[utterances_id] = REBED
    with open('data_HUB/HAE_REBED_dic.pkl'+str(data_type), 'wb') as f:
        pkl.dump(HAE_dic, f)




FLAGS = tf.flags.FLAGS

# write_sample(FLAGS.train_file, FLAGS.valid_file)

# with open('DSTC8_answer.pkl', 'wb') as f:
#     answer_pkl = pkl.load(f)
pasing_data(FLAGS.train_file, data_type='train')


pasing_data(FLAGS.valid_file, data_type='valid')
