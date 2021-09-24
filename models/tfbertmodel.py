

import argparse
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_ranking.extension import tfrbert
"""
This implementation is based on the guide provided in the TF Ranking library found here
https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/extension

"""
terminal = argparse.ArgumentParser()
  #define arguments
terminal.add_argument("--training_data", type=str, required=True, help="TFRecord file /path/to/file")
terminal.add_argument("--test_data", type=str, required=True, help="TFRecord file /path/to/file")
terminal.add_argument("--model_directory", type=str, required=True, help="txt file /path/to/file")
terminal.add_argument("--list_size", type=int, required=True, help="size 10 15")    
terminal.add_argument("--checkpoint_secs", type=int, required=True, help="seconds 100") 
terminal.add_argument("--learning_rate", type=float, required=True, help=" 0.005")
terminal.add_argument("--batch_size", type=int, required=True, help="size 8 16 32")  
terminal.add_argument("--max_seq_length", type=int, required=True, help="128")
terminal.add_argument("--training_steps", type=int, required=True, help="100000")
terminal.add_argument("--loss", type=str, required=True, help="1")
terminal.add_argument("--bert_config_file", type=str, required=True, help="config file /path/to/file")
terminal.add_argument("--bert_init_ckpt", type=str, required=True, help="directory with pretrained model /path/to/file")
terminal.add_argument("--num_eval_steps", type=int, required=True, help="10 steps")
terminal.add_argument("--dropout_rate", type=float, required=True, help="dropout for model")
terminal.add_argument("--num_checkpoints", type=int,required=True, help="100")

arguments = terminal.parse_args()


def context_feature_columns():
  """
  since bert uses [cls] query [sep] doc [sep] see example feature columns

  """
  return {}


def example_feature_columns():
  """
  example_feature_columns()- returns feature name to column definition.
  input_ids - feature id based on MAPPINGS from bert vocabularu
  input_mask - feature real or padded
  segment_ids - query or document

  returns dictionary containing column name definitons
  
  """
  feature_columns = {}
  feature_columns.update({
      "input_ids":
          tf.feature_column.numeric_column(
              "input_ids",
              shape=(arguments.max_seq_length,),
              default_value=0,
              dtype=tf.int64),
      "input_mask":
          tf.feature_column.numeric_column(
              "input_mask",
              shape=(arguments.max_seq_length,),
              default_value=0,
              dtype=tf.int64),
      "segment_ids":
          tf.feature_column.numeric_column(
              "segment_ids",
              shape=(arguments.max_seq_length,),
              default_value=0,
              dtype=tf.int64),
  })
  return feature_columns


def get_estimator(hparams):
  """
  get_estimator() - creates estimator framework for bert network

  returns the estimator
  """
  #utility functions for bert
  util = tfrbert.TFRBertUtil(
      bert_config_file=hparams.get("bert_config_file"),
      bert_init_ckpt=hparams.get("bert_init_ckpt"),
      bert_max_seq_length=hparams.get("bert_max_seq_length"))
  #defines bert network and sets appropriate parameters
  network = tfrbert.TFRBertRankingNetwork(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      bert_config_file=hparams.get("bert_config_file"),
      bert_max_seq_length=hparams.get("bert_max_seq_length"),
      bert_output_dropout=hparams.get("dropout_rate"),
      name="tfrbert")

  #define loss finction
  loss = tfr.keras.losses.get(
      hparams.get("loss"),
      reduction=tf.compat.v2.losses.Reduction.SUM_OVER_BATCH_SIZE)

  #metrics function
  metrics = tfr.keras.metrics.default_keras_metrics()

  #configuration for estimator
  config = tf.estimator.RunConfig(
      model_dir=hparams.get("model_dir"),
      keep_checkpoint_max=hparams.get("num_checkpoints"),
      save_checkpoints_secs=hparams.get("checkpoint_secs"))

  #optimizer set to adamw by defualt if not defined
  optimizer = util.create_optimizer(
      init_lr=hparams.get("learning_rate"),
      train_steps=hparams.get("num_train_steps"),
      warmup_steps=hparams.get("bert_num_warmup_steps"))

  #ranking head
  ranker = tfr.keras.model.create_keras_model(
      network=network,
      loss=loss,
      metrics=metrics,
      optimizer=optimizer,
      size_feature_name="example_list_size")

  #returns estimator
  return tfr.keras.estimator.model_to_estimator(
      model=ranker,
      model_dir=hparams.get("model_dir"),
      config=config,
      warm_start_from=util.get_warm_start_settings(exclude="tfrbert"))


def train_and_eval():
  """
    train_and_eval() - train and evaluate the model
  """
  #define hyper parameters for model
  hparams = dict(
      train_input_pattern=arguments.training_data,
      eval_input_pattern=arguments.test_data,
      learning_rate=arguments.learning_rate,
      train_batch_size=arguments.batch_size,
      eval_batch_size=arguments.batch_size,
      checkpoint_secs=arguments.checkpoint_secs,
      num_checkpoints= arguments.num_checkpoints,
      num_train_steps=arguments.training_steps,
      num_eval_steps=arguments.num_eval_steps,
      loss=arguments.loss,
      dropout_rate=arguments.dropout_rate,
      list_size=arguments.list_size,
      listwise_inference=True,
      convert_labels_to_binary=False,
      model_dir=arguments.model_directory,
      bert_config_file=arguments.bert_config_file,
      bert_init_ckpt=arguments.bert_init_ckpt,
      bert_max_seq_length=arguments.max_seq_length,
      bert_num_warmup_steps=10000)

  #ranking pipeline
  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
      context_feature_columns=context_feature_columns(),
      example_feature_columns=example_feature_columns(),
      hparams=hparams,
      estimator=get_estimator(hparams),
      label_feature_name="relevance",
      label_feature_type=tf.int64,
      size_feature_name="example_list_size")

  ranking_pipeline.train_and_eval(local_training=True)


def main(_):
  
  train_and_eval()


if __name__ == "__main__":
  tf.compat.v1.app.run()