import json
import random
import collections
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
import multiprocessing
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
    for index, label in enumerate(select_label):
        if label == "1":
            selected_utterances.append(utterances[index])


    # 2. Create the correct response
    # if len(dialog['options-for-correct-answers']) != 0:
    positives_id = []
    positives = []
    for correct in dialog['options-for-correct-answers']:
        positives_id.append(correct['candidate-id'])
        positives.append(correct['utterance'].lower())
    if len(dialog['options-for-correct-answers']) != 1 & len(dialog['options-for-correct-answers']) != 0:
        print(len(dialog['options-for-correct-answers']))
    if len(dialog['options-for-correct-answers']) == 1:
        flag = 1
    elif len(dialog['options-for-correct-answers']) == 0:
        flag = 0


    # 3. Create the response distractors
    if data_type == "train":
        num_distractors = 1
        if len(positives_id) == 1:
            positive_sample = {}
            positive_sample['candidate-id'] = dialog['options-for-correct-answers'][0]['candidate-id']
            positive_sample['utterance'] = dialog['options-for-correct-answers'][0]['utterance']
            dialog['options-for-next'].remove(positive_sample)
            # while dialog['options-for-correct-answers'][0] in dialog['options-for-next']:
            #     dialog['options-for-next'].remove(dialog['options-for-correct-answers'][0])
        distractors = random.sample(dialog['options-for-next'], num_distractors)      

    if data_type == "valid" or data_type == "test":
        if len(positives_id) == 1:
            positive_sample = {}
            positive_sample['candidate-id'] = dialog['options-for-correct-answers'][0]['candidate-id']
            positive_sample['utterance'] = dialog['options-for-correct-answers'][0]['utterance']
            dialog['options-for-next'].remove(positive_sample)
            # while dialog['options-for-correct-answers'][0] in dialog['options-for-next']:
            #     dialog['options-for-next'].remove(dialog['options-for-correct-answers'][0])
        distractors = dialog['options-for-next']

    # make sure that the correct response is not in the distractors
    # assert len([_ for _ in dialog['options-for-correct-answers'] if _ in distractors]) == 0

    negatives_id = []
    negatives = []
    for distractor in distractors:
        negatives_id.append(distractor['candidate-id'])
        negatives.append(distractor['utterance'].lower())

    return utterances_id, selected_utterances, positives_id, positives, negatives_id, negatives, flag


def data_parser(fname, data_type, epoch_id):

    if data_type == 'train':
        total_num = 112262
        filename = 'Ubuntu_dstc_{}_{}.txt'.format(data_type, epoch_id)
    elif data_type == 'valid':
        total_num = 9565
        filename = 'Ubuntu_dstc_{}.txt'.format(data_type)

    dataset_size = 0
    bad_sample = 0
    print("Generating the file of {} ...".format(filename))
    with open(filename, 'w') as fw:
        with open(fname, 'rb') as f:
            json_data = json.load(f)
            for i, entry in enumerate(json_data):
                if i % 1000 == 0:
                    print('Done {}'.format(i))

                us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, data_type)

                if flag == 0:
                    bad_sample += 1

                context = ' '.join(utterances)
                
                # write data to files
                for j in range(len(positives_id)):
                    dataset_size += 1
                    fw.write(str(us_id))
                    fw.write('\t')
                    fw.write(context)
                    fw.write('\t')
                    fw.write(positives_id[j])
                    fw.write('\t')
                    fw.write(positives[j])
                    fw.write('\t')
                    fw.write('follow')
                    fw.write('\n')

                for j in range(len(negatives_id)):
                    dataset_size += 1
                    fw.write(str(us_id))
                    fw.write('\t')
                    fw.write(context)
                    fw.write('\t')
                    fw.write(negatives_id[j])
                    fw.write('\t')
                    fw.write(negatives[j])
                    fw.write('\t')
                    fw.write('unfollow')
                    fw.write('\n')
            
            print("dataset_size: {}".format(dataset_size))
            print("bad_sample size: {}".format(bad_sample))

    return filename    


def write_answer(train_file, valid_file):

    answer_dic = {}

    with open(train_file, 'rb') as f:
        json_data = json.load(f)
        total_num = 112262
        for i, entry in tqdm(enumerate(json_data), total = total_num):
            us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, 'valid')
            for index, i in enumerate(positives_id):
                if i not in answer_dic:
                    answer_dic[i] = positives[index]
            for index, i in enumerate(negatives_id):
                if i not in answer_dic:
                    answer_dic[i] = negatives[index]

    with open(valid_file, 'rb') as f:
        json_data = json.load(f)
        total_num = 9565
        for i, entry in tqdm(enumerate(json_data), total = total_num):
            us_id, utterances, positives_id, positives, negatives_id, negatives, flag = process_dialog(entry, 'valid')
            for index, i in enumerate(positives_id):
                if i not in answer_dic:
                    answer_dic[i] = positives[index]
            for index, i in enumerate(negatives_id):
                if i not in answer_dic:
                    answer_dic[i] = negatives[index]


    with open('DSTC8_answer_OriginalID2utterance.pkl','wb') as f:
    	pkl.dump(answer_dic, f)

    # project the answer ID from string to int
    convert_dic = {}
    base = 70000000
    for index, item in tqdm(enumerate(answer_dic)):
    	convert_dic[item] = base + index

    with open('DSTC8_answer_OriginalID2NewID.pkl', 'wb') as f:
    	pkl.dump(convert_dic,f)

    return convert_dic





if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    train_files = []
    for epoch_id in range(FLAGS.num_train_epochs):
        filename = data_parser(FLAGS.train_file, 'train', epoch_id)
        train_files.append(filename)
    valid_file = data_parser(FLAGS.valid_file, 'valid', epoch_id)
    ans_dic = write_answer(FLAGS.train_file, FLAGS.valid_file)


