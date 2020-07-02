import json
import random
import collections
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
import multiprocessing
import tokenization
import ijson
import tokenization

tf.flags.DEFINE_string("train_file", "task-1.advising.train.json",
                       "path to train file")
tf.flags.DEFINE_string("valid_file", "task-1.advising.dev.json",
                       "path to valid file")

vocab_file = '../../../uncased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True


def process_dialog(dialog, data_type):
    """
    Add __eou__ and __eot__ tags between utterances
    create context, correct response and wrong responses.
    flag denotes whether having the correct response, 1: yes 0: no
    """

    # 1. Create the context (a list of utterances)
    utterances_id = dialog['example-id']
    context = dialog['messages-so-far']
    utterances = []
    utterances_speaker = []
    for i in range(len(context)):
        if i < (len(context) - 1):
            if context[i]['speaker'] == context[i + 1]['speaker']:
                utterances.append(context[i]['utterance'].lower() + " __eou__")
            else:
                utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        else:
            utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")

        utterances_speaker.append(context[i]['speaker'])

    assert len(utterances) == len(utterances_speaker)


    return utterances_id, utterances, utterances_speaker



def data_parser(fname, data_type):
    if data_type == 'train':
        total_num = 112262
    elif data_type == 'valid':
        total_num = 9565
    HAE_dic = {}
    dataset_size = 0
    bad_sample = 0
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    with open(fname, 'rb') as f:
        # json_data = json.load(f)
        json_data = ijson.items(f, 'item')
        for i, entry in enumerate(json_data):
            if i % 1000 == 0:
                print('Done {}'.format(i))

            utterances_id, utterances, speaker = process_dialog(entry,data_type)
            if len(speaker) ==0:
                bad_sample += 0

            sent = ' '
            context = sent.join(utterances)
            tokens = tokenizer.tokenize(context)
            REBED = [0] * len(tokens)
            for i, line in enumerate(speaker):
                if line == 'advisor':
                    index = context.find(utterances[i])
                    context1 = context[:index]
                    context_len = tokenizer.tokenize(context1)
                    response_len = tokenizer.tokenize(utterances[i])
                    for j in range(len(response_len)):
                        REBED[j + len(context_len)] = 1

            # if utterances_id in dic:
            HAE_dic[utterances_id] = REBED
        with open('data/HAE_REBED_dic.pkl' + str(data_type), 'wb') as f:
            pkl.dump(HAE_dic, f)
        print(bad_sample)


    # return filename



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    train_files = []
    filename = data_parser(FLAGS.train_file, 'train')
    valid_file = data_parser(FLAGS.valid_file, 'valid')
