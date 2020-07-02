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

tf.flags.DEFINE_string("train_file", "task-2.ubuntu.train.json",
                       "path to train file")
tf.flags.DEFINE_string("valid_file", "task-2.ubuntu.dev.json",
                       "path to valid file")
tf.flags.DEFINE_integer("num_train_epochs", 30,
                        "total number of training epochs to perform")
tf.flags.DEFINE_string("vocab_file", "../../../uncased_L-12_H-768_A-12/vocab.txt",
                       "path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 512,
                        "max sequence length of concatenated context and response")
tf.flags.DEFINE_bool("do_lower_case", True,
                     "whether to lower case the input text")


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
        # utterance
        if i < (len(context) - 1):
            if context[i]['speaker'] == context[i + 1]['speaker']:
                utterances.append(context[i]['utterance'].lower() + " __eou__")
            else:
                utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        else:
            utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        # utterances_speaker
        utterances_speaker.append(context[i]['speaker'])

    assert len(utterances) == len(utterances_speaker)

    # select relevant utterances
    next_speaker = dialog['next-speaker']
    combined_utterances_speaker = zip(utterances, utterances_speaker)
    select_label = []
    for (utterance, utterance_speaker) in combined_utterances_speaker:
        at_speaker = utterance.split()[0]
        if at_speaker == next_speaker or utterance_speaker == next_speaker:
            label = "1"
        else:
            label = "0"
        select_label.append(label)
    assert len(select_label) == len(utterances_speaker)

    selected_utterances = []
    selected_speaker = []
    selected_response = []
    for index, label in enumerate(select_label):
        if label == "1":
            selected_utterances.append(utterances[index])
            selected_speaker.append(utterances_speaker[index])

    for s_utterance, s_speaker in zip(selected_utterances, selected_speaker):
        if s_speaker != next_speaker:
            selected_response.append(s_utterance)





    return utterances_id, selected_utterances, selected_response


def data_parser(fname, data_type):
    if data_type == 'train':
        total_num = 112262
    elif data_type == 'valid':
        total_num = 9565
    HAE_dic = {}
    dataset_size = 0
    bad_sample = 0
    do_lower_case = True
    vocab_file = '../../uncased_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    with open(fname, 'rb') as f:
        # json_data = json.load(f)
        json_data = ijson.items(f, 'item')
        for i, entry in enumerate(json_data):
            if i % 1000 == 0:
                print('Done {}'.format(i))

            utterances_id, utterances, response = process_dialog(entry,data_type)
            if len(response) ==0:
                bad_sample += 0

            sent = ' '
            context = sent.join(utterances)
            tokens = tokenizer.tokenize(context)
            REBED = [0] * len(tokens)
            for line in response:
                index = context.find(line)
                context1 = context[:index]
                context_len = tokenizer.tokenize(context1)
                response_len = tokenizer.tokenize(line)
                for i in range(len(response_len)):
                    REBED[i + len(context_len)] = 1

            # if utterances_id in dic:
            HAE_dic[utterances_id] = REBED
        with open('data_HUB/HAE_REBED_dic.pkl' + str(data_type), 'wb') as f:
            pkl.dump(HAE_dic, f)
        print(bad_sample)


    # return filename



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    train_files = []
    filename = data_parser(FLAGS.train_file, 'train')
    valid_file = data_parser(FLAGS.valid_file, 'valid')
