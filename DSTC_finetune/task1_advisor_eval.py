# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import datetime
import argparse
import csv
import os
import Advisor_HAE_modeling as modeling
import optimization
import tokenization
import datetime
import random
import metrics
from collections import defaultdict
import tensorflow as tf
from time import time
import operator




flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "valid_dir", 'DSTC8_DATA/Task_1/ubuntu/data/DSTC_valid.tfrecord',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "restore_model_dir", 'adapted_L-12_H-768_A-12/bert_final',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'DSTC', "The name of the task to train.")

flags.DEFINE_string("vocab_file", 'uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", 'output/DSTC',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 320,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 12, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 12, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 5,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
#
# tf.flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")
#
# tf.flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
#
# tf.flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
#
# tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
#
# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
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
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example




def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, HAE_ids, labels,
                 num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      HAE_ids = HAE_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  target_loss_weight = [1.0, 1.0]
  target_loss_weight = tf.convert_to_tensor(target_loss_weight)

  flagx = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
  flagy = tf.cast(tf.equal(labels, 0), dtype=tf.float32)

  all_target_loss = target_loss_weight[1] * flagx + target_loss_weight[0] * flagy

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    output_layer = tf.layers.dropout(output_layer, rate=0.1, training=is_training)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    probabilities = tf.sigmoid(logits, name="prob")
    logits = tf.squeeze(logits,[1])
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = tf.multiply(losses, all_target_loss)

    mean_loss = tf.reduce_mean(losses, name="mean_loss") +  sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.sign(probabilities - 0.5), tf.sign(labels - 0.5))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
    #
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)

    return mean_loss, logits, probabilities, accuracy, model





def print_configuration_op(FLAGS):
    print('My Configurations:')
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')

def parse_exmp(serial_exmp):
    input_data = tf.parse_single_example(serial_exmp,
                                       features={
                                           "ques_ids":
                                               tf.FixedLenFeature([], tf.int64),
                                           "ans_ids":
                                               tf.FixedLenFeature([], tf.int64),
                                           "input_sents":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "HEA_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "label_ids":
                                               tf.FixedLenFeature([], tf.float32),

                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    ques_ids = input_data["ques_ids"]
    ans_ids = input_data['ans_ids']
    sents = input_data["input_sents"]
    msk = input_data["input_mask"]
    sids= input_data["segment_ids"]
    hae_ids = input_data["HEA_ids"]
    labels = input_data['label_ids']
    return ques_ids, ans_ids, sents, msk, sids, hae_ids, labels


def setDir(dir_name):
    dir_path = './' + str(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

best_score = 0.0
def run_test(epoch_no, dir_path,op_name, sess, training, accuracy, prob, pair_ids):
    results = defaultdict(list)
    num_test = 0
    num_correct = 0.0
    n_updates = 0
    mrr = 0
    t0 = time()
    try:
        while True:
            n_updates += 1

            batch_accuracy, predicted_prob, pair_ = sess.run([accuracy, prob, pair_ids],feed_dict={training:False})
            question_id, answer_id, label = pair_

            num_test += len(predicted_prob)

            num_correct += len(predicted_prob) * batch_accuracy
            for i, prob_score in enumerate(predicted_prob):
                # question_id, answer_id, label = pair_id[i]
                results[question_id[i]].append((answer_id[i], label[i], prob_score[0]))

            if n_updates%2000 == 0:
                tf.logging.info("epoch: %i  n_update %d , %s: Mins Used: %.2f" %
                                (epoch_no, n_updates, op_name, (time() - t0)/60.0 ))


    except tf.errors.OutOfRangeError:

        threshold = 0.95
        none_id = 10000000
        print("threshold: {}".format(threshold))
        for q_id, a_list in results.items():
            correct_flag = 0
            for (a_id, label, score) in a_list:
                if int(label) == 1:
                    correct_flag = 1
            if correct_flag == 0:
                results[q_id].append((none_id, 1, threshold))
            else:
                results[q_id].append((none_id, 0, threshold))
        # calculate top-1 precision
        print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct / num_test))
        accu, precision, recall, f1, loss = metrics.classification_metrics(results)
        print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

        mvp = metrics.mean_average_precision(results)
        mrr = metrics.mean_reciprocal_rank(results)
        top_1_precision = metrics.top_1_precision(results)
        total_valid_query = metrics.get_num_valid_query(results)
        print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(
            mvp, mrr, top_1_precision, total_valid_query))


        out_path = os.path.join(dir_path, "ubuntu_output_epoch_{}.txt".format(epoch_no))
        print("Saving evaluation to {}".format(out_path))
        with open(out_path, 'w') as f:
          f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
          for us_id, v in results.items():
            v.sort(key=operator.itemgetter(2), reverse=True)
            for i, rec in enumerate(v):
              r_id, label, prob_score = rec
              rank = i+1
              f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))

    return mrr



def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums

def print_weight(name):
    with open(name + str(random.randint(0, 100)), 'w') as fw:
        variables = tf.trainable_variables()
        for variable in variables:
            fw.write(variable.name)
            fw.write('\n')
            fw.write(str(variable.eval()))
            fw.write('\n')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    print_configuration_op(FLAGS)
    train_batch_size = FLAGS.train_batch_size
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    root_path = FLAGS.output_dir

    set = list(range(30))
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    ISOTIMEFORMAT = '%Y-%m-%d_%H:%M'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    root_path = os.path.join(root_path, FLAGS.task_name+theTime)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    log_dir = os.path.join(root_path,'log.txt')
    sample_num = total_sample(FLAGS.valid_dir)
    print('train_data {}', sample_num)


    filenames = tf.placeholder(tf.string, shape=[None])
    shuffle_size = tf.placeholder(tf.int64)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    # buffer_size 100
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(train_batch_size)
    iterator = dataset.make_initializable_iterator()

    training = tf.placeholder(tf.bool)

    ques_ids, ans_ids, sents, mask, segmentids, hae_ids, labels = iterator.get_next()  # output dir


    pair_ids = [ques_ids, ans_ids, labels]





    mean_loss, logits, probabilities, accuracy, model= create_model(bert_config,
                                                                       is_training=training,
                                                                       input_ids = sents,
                                                                       input_mask= mask,
                                                                       segment_ids= segmentids,
                                                                       HAE_ids = hae_ids,
                                                                       labels = labels,
                                                                       num_labels = 1,
                                                                       use_one_hot_embeddings = False)



    # init model with pre-training
    tvars = tf.trainable_variables()
    if FLAGS.init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,FLAGS.init_checkpoint)
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    valid_file = FLAGS.valid_dir
    restore_model_dir = FLAGS.restore_model_dir

    if FLAGS.do_train:
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())


            tf.logging.info("*** restore model ***")

            ckpt = tf.train.get_checkpoint_state(
                restore_model_dir)  # '/media/E/xiaoyu/repos/nli_bert/output/mrpc_output')
            variables = tf.trainable_variables()

            saver = tf.train.Saver(variables)
            saver.restore(sess, ckpt.model_checkpoint_path)


            tf.logging.info('Valid begin ')
            sess.run(iterator.initializer,
                feed_dict={filenames: [valid_file], shuffle_size: 1})
            run_test(0, root_path, "valid", sess, training, accuracy, probabilities, pair_ids)







if __name__ == "__main__":

  tf.app.run()

