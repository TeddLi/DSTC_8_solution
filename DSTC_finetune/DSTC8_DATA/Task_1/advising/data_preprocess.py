# import ijson
import json
import random
import collections
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
import multiprocessing
import tokenization
import os



tf.flags.DEFINE_string("train_file", "task-1.advising.train.json", 
	"path to train file")
tf.flags.DEFINE_string("valid_file", "task-1.advising.dev.json", 
	"path to valid file")
tf.flags.DEFINE_integer("num_train_epochs", 30,
	"total number of training epochs to perform")
tf.flags.DEFINE_string("vocab_file", "../../../uncased_L-12_H-768_A-12/vocab.txt", 
	"path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 320, 
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
    for i in range(len(context)):
        if i < (len(context) - 1):
            if context[i]['speaker'] == context[i + 1]['speaker']:
                utterances.append(context[i]['utterance'].lower() + " __eou__")
            else:
                utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")
        else:
            utterances.append(context[i]['utterance'].lower() + " __eou__ __eot__")


    # 2. Create the correct response
    positives_id = []
    positives = []
    if len(dialog['options-for-correct-answers']) == 1:
        correct = dialog['options-for-correct-answers'][0]
        positives_id.append(str(correct['candidate-id']))
        positives.append(correct['utterance'].lower())
        flag = 1
    elif len(dialog['options-for-correct-answers']) == 0:
        flag = 0


    # 3. Create the wrong responses
    if data_type == "train":
        num_distractors = 1
        if len(dialog['options-for-correct-answers']) == 1:
            # distractors is composed of (num_distractors*negetive)
            dialog['options-for-next'].remove(dialog['options-for-correct-answers'][0])
            distractors = random.sample(dialog['options-for-next'], num_distractors)
            # distractors.append(none_response)
        elif len(dialog['options-for-correct-answers']) == 0:
            # distractors is composed of (num_distractors*negetive)
            distractors = random.sample(dialog['options-for-next'], num_distractors)

    if data_type == "valid" or data_type == "test":
        if len(dialog['options-for-correct-answers']) == 1:
            # distractors is composed of (99*negetive)
            dialog['options-for-next'].remove(dialog['options-for-correct-answers'][0])
            distractors = dialog['options-for-next']
            # distractors.append(none_response)
        elif len(dialog['options-for-correct-answers']) == 0:
            # distractors is composed of (100*negetive)
            distractors = dialog['options-for-next']

    negatives_id = []
    negatives = []
    for distractor in distractors:
        negatives_id.append(str(distractor['candidate-id']))
        negatives.append(distractor['utterance'].lower())

    return utterances_id, utterances, positives_id, positives, negatives_id, negatives, flag


def data_parser(fname, data_type, epoch):

    if data_type == 'train':
        total_num = 100000
        filename = 'Advising_dstc_{}_{}.txt'.format(data_type, epoch)
    elif data_type == 'valid':
        total_num = 500
        filename = 'Advising_dstc_{}.txt'.format(data_type)

    dataset_size = 0
    bad_sample = 0
    print("Generating the file of {} ...".format(filename))
    with open(os.path.join(filename), 'w') as fw:
        with open(fname, 'rb') as f:
            # json_data = ijson.items(f, 'item')
            json_data = json.load(f)
            for i, entry in enumerate(json_data):
                if i % 100 == 0:
                    print('Done {}'.format(i))

                us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, data_type)

                if flag == 0:
                    bad_sample += 1

                context = ' '.join(utterances)
                
                # write data to files
                for j in range(len(positives_id)):
                    dataset_size += 1
                    to_write_list = [str(us_id),  " ".join(context.split("\n")), positives_id[j], " ".join(positives[j].split("\n")), 'follow']
                    fw.write("\t".join(to_write_list))
                    fw.write('\n')
                    # fw.write(str(us_id))
                    # fw.write('\t')
                    # fw.write(context)
                    # fw.write('\t')
                    # fw.write(positives_id[j])
                    # fw.write('\t')
                    # fw.write(positives[j])
                    # fw.write('\t')
                    # fw.write('follow')
                    # fw.write('\n')

                for j in range(len(negatives_id)):
                    dataset_size += 1
                    to_write_list = [str(us_id), " ".join(context.split("\n")), negatives_id[j], " ".join(negatives[j].split("\n")), 'unfollow']
                    fw.write("\t".join(to_write_list))
                    fw.write('\n')
                    # fw.write(str(us_id))
                    # fw.write('\t')
                    # fw.write(context)
                    # fw.write('\t')
                    # fw.write(negatives_id[j])
                    # fw.write('\t')
                    # fw.write(negatives[j])
                    # fw.write('\t')
                    # fw.write('unfollow')
                    # fw.write('\n')
            
            print("dataset_size: {}".format(dataset_size))
            print("bad_sample size: {}".format(bad_sample))

    return filename    


def write_answer(train_file, valid_file):

    answer_dic = {}

    with open(train_file, 'rb') as f:
        # json_data = ijson.items(f, 'item')
        json_data = json.load(f)
        total_num = 100000
        for i, entry in tqdm(enumerate(json_data), total = total_num):
            us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, 'valid')
            for index, i in enumerate(positives_id):
                if i not in answer_dic:
                    answer_dic[i] = positives[index]
            for index, i in enumerate(negatives_id):
                if i not in answer_dic:
                    answer_dic[i] = negatives[index]

    with open(valid_file, 'rb') as f:
        # json_data = ijson.items(f, 'item')
        json_data = json.load(f)
        total_num = 500
        for i, entry in tqdm(enumerate(json_data), total = total_num):
            us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, 'valid')
            for index, i in enumerate(positives_id):
                if i not in answer_dic:
                    answer_dic[i] = positives[index]
            for index, i in enumerate(negatives_id):
                if i not in answer_dic:
                    answer_dic[i] = negatives[index]

    with open('data/DSTC8_answer_OriginalID2utterance.pkl','wb') as f:
    	pkl.dump(answer_dic, f)


    # project the answer ID from string to int
    convert_dic = {}
    base = 70000000
    for index, item in tqdm(enumerate(answer_dic)):
    	convert_dic[item] = base + index

    with open('data/DSTC8_answer_OriginalID2NewID.pkl', 'wb') as f:
    	pkl.dump(convert_dic,f)

    return convert_dic


if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    train_files = []
    for epoch in range(FLAGS.num_train_epochs):
        filename = data_parser(FLAGS.train_file, 'train', epoch)
        train_files.append(filename)
    valid_file = data_parser(FLAGS.valid_file, 'valid', epoch)
    ans_dic = write_answer(FLAGS.train_file, FLAGS.valid_file)



