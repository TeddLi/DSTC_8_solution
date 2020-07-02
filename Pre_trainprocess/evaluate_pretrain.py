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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import HAE_modeling as modeling
import optimization
import tensorflow as tf
from time import time
import datetime
# import tensorflow.contrib.data as dataset


#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer(
    "sample_num", '126',
    "total sample number")

flags.DEFINE_integer(
    "mid_save_step", '3000',
    "Epoch is so long, mid_save_step 15000 is roughly 3 hours")

flags.DEFINE_string(
    "input_file", 'output/manual_external',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "restore_dir", 'output/bert_model_2019-08-25-22:00:42_epoch_0.7563457407650185',' restore_model')

flags.DEFINE_string(
    "bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'adaptation', "The name of the task to train.")

flags.DEFINE_string("vocab_file", 'uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", './adapted_L-12_H-768_A-12',
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 25,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

# flags.DEFINE_integer("num_train_steps", 600, "Number of training steps. In our case, 4800000 / batchsize * epoch,")

flags.DEFINE_float("warmup_proportion", 0.1, "Number of warmup steps.")

flags.DEFINE_integer("num_train_epochs", 10, "num_train_epochs.")

# flags.DEFINE_integer("save_checkpoints_steps", 1000,
#                      "How often to save the model checkpoint.")
#
# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")
#
# flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")
#
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


def model_fn_builder(features, Is_training, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    input_ids, input_mask, segment_ids, REBED_ids, masked_lm_positions, \
    masked_lm_ids, masked_lm_weights, next_sentence_labels = features

    is_training = Is_training

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        HAE_ids=REBED_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs, labels) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)


    matrix = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels)

    return train_op, total_loss, matrix, labels


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }





def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs,labels)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""




def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example



def run_epoch( epoch, sess, lm_losses, saver, root_path, save_step, mid_save_step, phase, batch_size=16, train_op=tf.constant(0)):
    t_loss = 0
    n_all = 0
    t0 = time()
    t1 = time()


    step = 0

    print('running begin ... ')
    try:
        while True:
            step = step + 1
            batch_loss, _ = sess.run([lm_losses, train_op] )


            n_sample = batch_size
            n_all += n_sample
            t_loss += batch_loss * n_sample
            # save every epoch or 3 hour
            # if (step % save_step == 0) or (step % 15000 == 0):
            if (step % mid_save_step == 0):
                c_time = str(datetime.datetime.now()).replace(' ', '-').split('.')[0]
                save_path = os.path.join(root_path, 'bert_model_{0}_epoch_{1}'.format(c_time, epoch))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, os.path.join(save_path,'bert_model_{}.ckpt'.format(c_time)), global_step = step)
                print('save model epoch {}'.format(int(step/save_step)))
                print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
                      format(phase, t_loss / n_all, time() - t1))
                t1=time()
                print('Sample seen {} total time {}'.format(n_all,time() - t0))
    except tf.errors.OutOfRangeError:
        print('Epoch {} Done'.format(epoch))
        c_time = str(datetime.datetime.now()).replace(' ', '-').split('.')[0]
        save_path = os.path.join(root_path, 'bert_model_{0}_epoch_{1}'.format(c_time, step / save_step))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(sess, os.path.join(save_path, 'bert_model_{}.ckpt'.format(c_time)), global_step=step)
        print('save model epoch {}'.format(int(step / save_step)))
        print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
              format(phase, t_loss / n_all, time() - t1))
        t1 = time()
        print('Sample seen {} total time {}'.format(n_all, time() - t0))
        pass




def test_epoch( sess, lm_losses, evaluate, eval_op, labels, root_path, save_step, mid_save_step, phase, batch_size=16, train_op=tf.constant(0)):
    t_loss = 0
    n_all = 0
    t0 = time()
    t1 = time()


    masked_lm_accuracy = 0.0
    masked_lm_mean_loss = 0.0
    next_sentence_accuracy = 0.0
    next_sentence_mean_loss = 0.0


    step = 0

    print('running begin ... ')
    try:
        while True:
            step = step + 1
            y, matrix, batch_loss, _, _ = sess.run([ labels, evaluate, lm_losses, train_op, eval_op] )

            masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss = matrix

            # masked_lm_accuracy = evaluate["masked_lm_accuracy"]
            # masked_lm_mean_loss = evaluate["masked_lm_loss"]
            # next_sentence_accuracy = evaluate["next_sentence_accuracy"]
            # next_sentence_mean_loss = evaluate["next_sentence_loss"]



            n_sample = len(y)
            n_all += n_sample
            # save every epoch or 3 hour
            # if (step % save_step == 0) or (step % 15000 == 0):
            if (step % mid_save_step == 0):

                print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
                      format(phase, t_loss / n_all, time() - t1))

                print('masked_lm_accuracy  {:.6f},  masked_lm_mean_loss {:.6f},  next_sentence_accuracy {:.6f},  next_sentence_mean_loss{:.6f}'.format(
                    masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss
                ))

                t1=time()
                print('Sample seen {} total time {}'.format(n_all,time() - t0))

    except tf.errors.OutOfRangeError:

        print('save model epoch {}'.format(int(step / save_step)))
        print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
              format(phase, t_loss / n_all, time() - t1))
        t1 = time()
        print('Sample seen {} total time {}'.format(n_all, time() - t0))
        pass






def parse_exmp(serial_exmp):
    input_data = tf.parse_single_example(serial_exmp,
                                       features={
                                           "input_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "REBED_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "masked_lm_positions":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_ids":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
                                           "next_sentence_labels":
                                               tf.FixedLenFeature([1], tf.int64),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    ids = input_data["input_ids"]
    msk = input_data["input_mask"]
    sids= input_data["segment_ids"]
    REBED_ids = input_data["REBED_ids"]
    m_lp = input_data["masked_lm_positions"]
    m_lids = input_data["masked_lm_ids"]
    m_lm_w = input_data["masked_lm_weights"]
    nsl = input_data["next_sentence_labels"]
    return ids, msk, sids, REBED_ids, m_lp, m_lids, m_lm_w, nsl


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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print_configuration_op(FLAGS)
    train_batch_size = FLAGS.train_batch_size
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    root_path = FLAGS.output_dir
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cd = os.path.join(root_path,FLAGS.task_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)


    sample_num = FLAGS.sample_num

    num_train_steps = sample_num // train_batch_size * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    # filename_queue = tf.train.string_input_producer([FLAGS.input_file], num_epochs=None)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)


    # dataset = tf.data.TFRecordDataset(FLAGS.input_file)

    buffer_size = 1000
    # dataset = tf.data.TFRecordDataset(FLAGS.input_file)
    # dataset = dataset.map(parse_exmp)
    # # FLAGS.num_train_epochs
    # dataset = dataset.repeat(1).shuffle(buffer_size).batch(train_batch_size)

    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(1)
    dataset = dataset.batch(train_batch_size)
    iterator = dataset.make_initializable_iterator()
    save_step = sample_num // train_batch_size

    # iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    # input_set=[ids, msk, sids, m_lp, m_lids, m_lm_w, nsl]


    input_ids, input_mask, segment_ids,  REBED_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights,next_sentence_labels = iterator.get_next()
    features = [input_ids, input_mask, segment_ids,  REBED_ids,masked_lm_positions, masked_lm_ids, masked_lm_weights,
                next_sentence_labels]
    train_op, loss, matrix, labels = model_fn_builder(
        features,  # ----model_fn_builder----
        Is_training= False,
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps= num_train_steps,
        num_warmup_steps= num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)


    masked_lm_accuracy, masked_acc_op = matrix["masked_lm_accuracy"]
    masked_lm_mean_loss, masked_loss_op= matrix["masked_lm_loss"]
    next_sentence_accuracy, next_sentence_op = matrix["next_sentence_accuracy"]
    next_sentence_mean_loss, next_sentence_loss_op = matrix["next_sentence_loss"]

    evaluate = [ masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss]
    eval_op = [ masked_acc_op, masked_loss_op, next_sentence_op, next_sentence_loss_op]



    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    model_dir = FLAGS.restore_dir
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_dir)  # '/media/E/xiaoyu/repos/nli_bert/output/mrpc_output')
        if ckpt:
            variables = tf.trainable_variables()
            # variables_to_resotre = [v for v in variables if v.name.split('/')[0] == 'bert']
            # variables_to_resotre = [v for v in variables_to_resotre if v.name.split('/')[1] != 'pooler']
            tf.logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver = tf.train.Saver(variables)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.input_file]})
        test_epoch(sess, loss, evaluate, eval_op, labels, root_path, save_step, FLAGS.mid_save_step,'train', batch_size=train_batch_size)



if __name__ == "__main__":
    tf.app.run()


