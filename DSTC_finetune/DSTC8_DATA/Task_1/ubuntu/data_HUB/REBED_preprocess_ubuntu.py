
import os
import pickle
import numpy as np
import pickle as pkl
import tokenization
import csv
from tqdm import tqdm
import tensorflow as tf
import collections
import multiprocessing
from multiprocessing import pool



with open('../DSTC8_answer_OriginalID2NewID.pkl','rb') as f:
    ans_dic = pkl.load(f)

with open('HAE_REBED_dic.pkltrain', 'rb') as f:
    train_dic = pkl.load(f)
with open('HAE_REBED_dic.pklvalid', 'rb') as f:
    val_dic = pkl.load(f)
def read_sentence(path):

    with open(path, encoding='utf-8', errors='ignore') as f:
        sent = []
        for line in f:
            sent.append(line.replace('\n', ''))
        return sent

class InputExample(object):
    def __init__(self, guid,ques_ids, text_a, ans_ids, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, ques_ids, ans_ids, input_sents, input_mask, segment_ids, HEA_ids, label_id, tokens):
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.HEA_ids = HEA_ids
        self.label_id = label_id
        self.tokens = tokens


def read_DSTC(input_file):
    lines = []
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            concat = []
            temp = line.rstrip().split('\t')
            concat.append(temp[0]) # contxt id
            concat.append(temp[1]) # contxt
            concat.append(ans_dic[temp[2]]) # ans id
            concat.append(temp[3]) # ans
            concat.append(temp[4]) # label
            lines.append(concat)


        return lines
# def read_dev_txt(input_file):
#     lines = []
#     with open(input_file, 'r') as f:
#         for line in f:
#             concat = []
#             temp = line.rstrip().split('\t')
#             concat.append()
#             lines.append(temp)
#         return line
def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    dict = {'0': 'unfollow', '1': 'follow'}
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        ques_ids = line[0]
        # print("---line[8]", line[8])
        text_a = tokenization.convert_to_unicode(line[1])
        ans_ids = line[2]
        # print("---tex_a", text_a)
        text_b = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(InputExample(guid=guid, ques_ids = ques_ids, text_a=text_a, ans_ids=ans_ids, text_b=text_b, label=label))
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, HAE_dic):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i
    over_number = 0
    hae_number = 0
    total_number = 0
    features = []  # feature
    total_number = len(examples)
    # for (ex_index, example) in tqdm(enumerate(examples),total=len(examples)):
    for (ex_index, example) in enumerate(examples):

        ques_ids = int(example.ques_ids)
        ans_ids = int(example.ans_ids)
        tokens_a = tokenizer.tokenize(example.text_a)  # text_a tokenize

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  # text_b tokenize

        if tokens_b:  # if has b
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # truncate
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because  # (?)
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        HEBED = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        HEBED.append(0)
        Re_index = 0
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
            HEBED.append(int(HAE_dic[ques_ids][Re_index]))
            Re_index += 1
            # if token in HAE_dic[ques_ids]:
            #     HAE.append(1)
            # else:
            #     HAE.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        HEBED.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                HEBED.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)
            HEBED.append(0)

        input_sents = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_sents)  # mask

        if len(input_sents) > max_seq_length:
            input_sents = input_sents[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
            segment_ids = segment_ids[:max_seq_length]
            HEBED=HEBED[:max_seq_length]
            over_number+=1


        # Zero-pad up to the sequence length.
        while len(input_sents) < max_seq_length:
            input_sents.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            HEBED.append(0)

        assert len(input_sents) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(HEBED) == max_seq_length

        label_id = label_map[example.label]


        if ex_index%2000==0:
            print('finish {} samples'.format(ex_index))

        # if ex_index < 5:  # output
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("guid: %s" % (example.guid))
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in tokens]))
        #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #   tf.logging.info(
        #       "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(  # object
                ques_ids=ques_ids,
                ans_ids = ans_ids,
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                HEA_ids = HEBED,
                label_id=label_id,
                tokens = tokens
            ))
    print(over_number)
    print(hae_number/total_number)
    return features

def feature_list(features):
    # ids, mask, sgid, label = [],[],[],[]
    # for i in range(len(features)):
    #     ids.append(features[i].input_ids)
    #     mask.append(features[i].input_mask)
    #     sgid.append(features[i].segment_ids)
    #     label.append(features[i].label_id)
    # return ids, mask, sgid, label
    l = len(features)
    ques_ids = np.zeros(shape=(l))
    ans_ids = np.zeros(shape=(l))
    sents = np.zeros(shape=(l, max_seq_length))
    mask = np.zeros(shape=(l, max_seq_length))
    sgid = np.zeros(shape=(l, max_seq_length))
    label = np.zeros(shape=(l))
    for i in range(len(features)):
        ques_ids[i] = features[i].ques_ids
        ans_ids[i] = features[i].ans_ids
        sents[i] = features[i].input_sents
        mask[i] = features[i].input_mask
        sgid[i] = features[i].segment_ids
        label[i] = features[i].label_id
    return ques_ids, ans_ids, sents, mask, sgid, label


#
# def write_instance_to_example_files(instances, tokenizer, max_seq_length,
#                                     max_predictions_per_seq, output_files):
#   """Create TF example files from `TrainingInstance`s."""
#   writers = []
#   for output_file in output_files:
#     writers.append(tf.python_io.TFRecordWriter(output_file))
#
#   writer_index = 0
#
#   total_written = 0
#   for (inst_index, instance) in enumerate(instances):
#     input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
#     input_mask = [1] * len(input_ids)
#     segment_ids = list(instance.segment_ids)
#     assert len(input_ids) <= max_seq_length
#
#     while len(input_ids) < max_seq_length:
#       input_ids.append(0)
#       input_mask.append(0)
#       segment_ids.append(0)
#
#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length
#
#     masked_lm_positions = list(instance.masked_lm_positions)
#     masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
#     masked_lm_weights = [1.0] * len(masked_lm_ids)
#
#     while len(masked_lm_positions) < max_predictions_per_seq:
#       masked_lm_positions.append(0)
#       masked_lm_ids.append(0)
#       masked_lm_weights.append(0.0)
#
#     next_sentence_label = 1 if instance.is_random_next else 0
#
#     features = collections.OrderedDict()
#     features["input_ids"] = create_int_feature(input_ids)
#     features["input_mask"] = create_int_feature(input_mask)
#     features["segment_ids"] = create_int_feature(segment_ids)
#     features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
#     features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
#     features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
#     features["next_sentence_labels"] = create_int_feature([next_sentence_label])
#
#     tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#
#     writers[writer_index].write(tf_example.SerializeToString())
#     writer_index = (writer_index + 1) % len(writers)
#
#     total_written += 1
#
#     if inst_index < 20:
#       tf.logging.info("*** Example ***")
#       tf.logging.info("tokens: %s" % " ".join(
#           [tokenization.printable_text(x) for x in instance.tokens]))
#
#       for feature_name in features.keys():
#         feature = features[feature_name]
#         values = []
#         if feature.int64_list.value:
#           values = feature.int64_list.value
#         elif feature.float_list.value:
#           values = feature.float_list.value
#         tf.logging.info(
#             "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
#
#   for writer in writers:
#     writer.close()
#
#   tf.logging.info("Wrote %d total instances", total_written)


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        # pair_ids[i] = features[i].input_ids
        # sents[i] = features[i].input_sents
        # mask[i] = features[i].input_mask
        # sgid[i] = features[i].segment_ids
        features = collections.OrderedDict()
        features["ques_ids"] = create_int_feature([instance.ques_ids])
        features["ans_ids"] = create_int_feature([instance.ans_ids])
        features["input_sents"] = create_int_feature(instance.input_sents)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features['HEA_ids'] = create_int_feature(instance.HEA_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written+=1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
    print("Wrote %d total instances", total_written)
    for writer in writers:
        writer.close()
    tf.logging.info("Wrote %d total instances", total_written)

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

global worker_id
def worker(input_name):
    #print('Worker id {}', worker_id)
    #worker_id +=1
    train_examples = create_examples(
        read_DSTC( os.path.join(input_dir, "Ubuntu_dstc_train_" + str(input_name) + '.txt')),
        "train")  # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_dic)
    write_instance_to_example_files(train_features, [ os.path.join(output_dir,'Ubuntu_REB_dstc_train_'+str(input_name)+'.tfrecord')])
    print('Done {}'.format(input_name))


if __name__ == "__main__":


    # config
    # fig = Config()
    tf.logging.set_verbosity(tf.logging.INFO)
    vocab_file          = '../../../../uncased_L-12_H-768_A-12/vocab.txt'           # "uncased_L-12_H-768_A-12/vocab.txt"
    input_dir            = "../"          # "KB/data_SNLI_1"
    output_dir          = './'
    max_seq_length      = 320        # 128
    do_lower_case       = True        # True
    num_epoch           = 30
    label_list          = ["unfollow", "follow"]

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    worker_id = 0


    print("***** Read train_data *****")
    #pool = multiprocessing.Pool(processes=10)
    result = []
    list_set = list(range(num_epoch))
    # for input_name in [2518, 2570, 3923, 4189, 4450, 5398, 6222, 9625, 980]:
    for input_name in list_set:
        #msg = "input_name %d" % (input_name)
        #result.append(pool.apply_async(worker, (input_name,)))
    #pool.close()
    #pool.join()


        p = multiprocessing.Process(target=worker, args=(input_name,))
        result.append(p)
        p.start()
        p.join()
        # result.append(pool.apply_async(worker, (input_name,)))
    print("Sub-process(es) done.")

        # train_examples = create_examples(
        #         read_txt(os.path.join(input_dir, "Ubuntu_dstc_train" + str(input_name))),
        #         "train")  # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
        #     train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
        #     write_instance_to_example_files(train_features, [output_dir + '_train.tfrecord' + str(input_name)])

    # train_examples = create_examples(read_txt(os.path.join(input_dir, "train.txt")), "train")  # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    # train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    # write_instance_to_example_files(train_features, [output_dir+'_train'])
    # train_ids, train_sents, train_mask, train_sgid, train_label = feature_list(train_features)
    # print("train:", len(train_sents))

    print("***** Read dev_data *****")
    dev_examples = create_examples(read_DSTC(os.path.join(input_dir, "Ubuntu_dstc_valid.txt")), "dev")         # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    dev_features = convert_examples_to_features(dev_examples, label_list, max_seq_length, tokenizer, val_dic)
    write_instance_to_example_files(dev_features, [ os.path.join(output_dir,'Ubuntu_REB_dstc_valid.tfrecord')])



