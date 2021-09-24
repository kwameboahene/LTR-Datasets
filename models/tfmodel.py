
import argparse
import json
import tensorflow as tf
import tensorflow_ranking as tfr
"""
this implementation is a based on the guide TF-ranking implementation found http://dx.doi.org/10.1145/3292500.3330677
 this implementation is also based on the example guide found in TensorFlow Library here
https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_tfrecord.py

"""
def main():

  
  #define arguments
  terminal = argparse.ArgumentParser()
  
  terminal.add_argument("--training_data", type=str, required=True, help="TFRecord file /path/to/file")
  terminal.add_argument("--test_data", type=str, required=True, help="TFRecord file /path/to/file")
  terminal.add_argument("--model_directory", type=str, required=True, help="txt file /path/to/file")
  terminal.add_argument("--vocab_file", type=str, required=True, help="txt file /path/to/file")
  terminal.add_argument("--list_size", type=int, required=True, help="size 10 15")    
  terminal.add_argument("--feature_label", type=str, required=True, help="Relevance") 
  terminal.add_argument("--learning_rate", type=float, required=True, help=" 0.005")
  terminal.add_argument("--batch_size", type=int, required=True, help="size 8 16 32")  
  terminal.add_argument("--dropout_rate", type=float, required=True, help=" 0.3-08.8 15")
  terminal.add_argument("--embedding_dimension", type=int, required=True, help="10 20")
  terminal.add_argument("--training_steps", type=int, required=True, help="100000")
  terminal.add_argument("--group_size", type=int, required=True, help="1")



  arguments = terminal.parse_args()

  def context_feature_columns():
      """
      example_feature_columns() - takes a query features and embeds them based on
      dimension speciifed and vocabulary  

      returns: embedded qeury features 
      """
      #feature column each word in vocabulary is represented
      sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
          key="query",
          vocabulary_file=arguments.vocab_file)
      #feature column for query embedding
      query_embedding_column = tf.feature_column.embedding_column(
          sparse_column, arguments.embedding_dimension)
      
      return {"query": query_embedding_column}


  def example_feature_columns():
      """
      example_feature_columns() - takes a document features and embeds them based on
      dimension speciifed and vocabulary  

      returns: Embedded document features 
      """
      #feature column each word in vocabulary is represented
      sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
          key="Document",
          vocabulary_file=arguments.vocab_file)
      #feature column for query embedding
      document_embedding_column = tf.feature_column.embedding_column(
          sparse_column, arguments.embedding_dimension)
      
      return {"Document": document_embedding_column}


  def input_fn(path, num_epochs=None):
      """
      input_fn(path, num_epochs=None) - builds tfr dataset

      parameter =  path- tfrecord containing query-document pairs
                  number_epochs - number of iterations on data
      
      returns features and their labels
      based on guide provided https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_tfrecord.py
      """
      #returns a dict mapping each feature key to data type[bytes,int64 or float]
      context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
      #feature column for label is defined 
      label_column = tf.feature_column.numeric_column(
          arguments.feature_label, dtype=tf.int64, default_value=-1)
      #returns a dict mapping each feature key to data type[bytes,int64 or float]
      example_feature_spec = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns().values()) + [label_column])

      # builds ranking dataset 
      dataset = tfr.data.build_ranking_dataset(
          file_pattern=path,
          data_format=tfr.data.ELWC,
          batch_size=arguments.batch_size,
          list_size=arguments.list_size,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec,
          reader=tf.data.TFRecordDataset,
          shuffle=False,
          num_epochs=num_epochs)
      #an iterator for each element in dataset
      features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
      #gets label feature value- removes dimension too by 1
      label = tf.squeeze(features.pop(arguments.feature_label), axis=2)
      #cast a tensor of different datatype float32
      label = tf.cast(label, tf.float32)
      return features, label

  def make_transform_fn():
    """
      signature function for transform function

    """ 
    def _transform_fn(features, mode):
      """
          transform_fn() - encodes context and example features 

          parameters- features
          mode = indicates whether we are training or evaluating our model

      returns: Embedded document features 
    
      """
        # feature columns are transformed from sparse to denser columns
      context_features, example_features = tfr.feature.encode_listwise_features(
              features=features,
              context_feature_columns=context_feature_columns(),
              example_feature_columns=example_feature_columns(),
              mode=mode,
              scope="transform_layer")

      return context_features, example_features

    return _transform_fn


  def make_score_fn():
    """
    signature function for score function
    """

    def _score_fn(context_features, group_features, mode, params, config):
          
      """
      _score_fn() - takes context and example features and scores each document

          parameters-
                  context_features -context features
                  group_features - example features
                  mode - eval predict, or training
                  hyperparameters for NN

          # adopted from https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_tfrecord.py

          returns scores
      """
          #hidden layers
      hidden_layers = ["64", "32", "16"]
          #we flatten all features inputs tensors and reshapes them to a shape equal to the number of documents in the tensor 
      with tf.compat.v1.name_scope("input_layer"):
        context_input = [
                  tf.compat.v1.layers.flatten(context_features[name])
                  for name in sorted(context_feature_columns())
        ]
        group_input = [
                  tf.compat.v1.layers.flatten(group_features[name])
                  for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(context_input + group_input, 1)
        #mode set to training
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        current_layer = input_layer
        #perform batch normalization
        current_layer = tf.compat.v1.layers.batch_normalization(
              current_layer,
              training=is_training,
              momentum=0.99)

        for i, layer_width in enumerate(int(d) for d in hidden_layers):
          current_layer = tf.compat.v1.layers.dense(current_layer, units=layer_width)
              #perform batch normalization
          current_layer = tf.compat.v1.layers.batch_normalization(
                  current_layer,
                  training=is_training,
                  momentum=0.99)
              #activation function set to RELU
          current_layer = tf.nn.relu(current_layer)
          #set dropout 
          current_layer = tf.compat.v1.layers.dropout(
                  inputs=current_layer, rate=arguments.dropout_rate, training=is_training)
          #scores
          logits = tf.compat.v1.layers.dense(current_layer, units=1)

      return logits

    return _score_fn


  def eval_metric_fns():
    
    """
            eval_metric_fns() -  

            parameters- features
            mode = indicates whether we are training or evaluating our model

        returns: dictinary of metric functions
      
    """
    #create dictionary for all metrics 
    metric_fns = {}
    #update to include metrics we want
    metric_fns.update({
          "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
              tfr.metrics.RankingMetricKey.MRR,
              tfr.metrics.RankingMetricKey.MAP,

          ]
    })
    metric_fns.update({
          "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
              tfr.metrics.RankingMetricKey.NDCG, topn=topn)
          for topn in [1, 3, 5, 10]
    })

    return metric_fns

  #set loss function 
  _LOSS = tfr.losses.RankingLossKey.SOFTMAX_LOSS
  loss_fn = tfr.losses.make_loss_fn(_LOSS)

  #set ranking optimizer
  optimizer = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=arguments.learning_rate)

  def _train_op_fn(loss):
    """
    defines training operations used in ranking head
    parameter - loss function

    returns- operation head
    """

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    minimize_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())

    train_op = tf.group([update_ops, minimize_op])

    return train_op







  def train_and_eval_fn():

    """
      train_and_eval_fn() - creates estimator and defines training
      and evaluation operations parameters

      returns estimator and parameters for training and evaluation 
    """
    #configuration for saving monitoring learning
    run_config = tf.estimator.RunConfig(
          save_checkpoints_steps=1000)
      #create estimator
    ranker = tf.estimator.Estimator(
        model_fn= model_fn,
        model_dir=arguments.model_directory,
        config=run_config)
    #build dataset for training
    train_input_fn = lambda: input_fn(arguments.training_data)
    #build dataset for evaluation
    eval_input_fn = lambda: input_fn(arguments.test_data, num_epochs=1)
    #build estimator for training
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=arguments.training_steps)
    #build estimator for evaluation
    eval_spec = tf.estimator.EvalSpec(
        name="eval",
        input_fn=eval_input_fn,
        throttle_secs=15)
    
    return (ranker, train_spec, eval_spec)


  #ranking head for loss and metrics  
  ranking_head = tfr.head.create_ranking_head(
        loss_fn=loss_fn,
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

  #model builder
  model_fn = tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            transform_fn=make_transform_fn(),
            group_size=1,
            ranking_head=ranking_head)
  #run model
  ranker, train_spec, eval_spec = train_and_eval_fn()
  #log training and evaluation process
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
 


if __name__ == "__main__":
  main()



