import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

_CATEGORICAL_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'species'


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')


def _fill_in_missing(x):
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def preprocessing_fn(inputs):
    outputs = {}
    
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])
    
    outputs[_transformed_name(_LABEL_KEY)] = _fill_in_missing(inputs[_LABEL_KEY])
    
    return outputs


def _input_fn(filenames, tf_transform_output, batch_size):   
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    # dataset = tf.contrib.data.make_batched_features_dataset(filenames, batch_size, transformed_feature_spec,
    #                                                              reader=tf.data.TFRecordDataset)
    dataset = tf.data.experimental.make_batched_features_dataset(
        filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)
    
    transformed_features = dataset.make_one_shot_iterator().get_next()
    
    transformed_labels = transformed_features.pop(_transformed_name(_LABEL_KEY))
    
    return transformed_features, transformed_labels


def _eval_input_receiver_fn(tf_transform_output, schema):
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')

    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    transformed_features = tf_transform_output.transform_raw_features(features)

    receiver_tensors = {'examples': serialized_tf_example}

    features.update(transformed_features)

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_transformed_name(_LABEL_KEY)])


def _example_serving_receiver_fn(tf_transform_output, schema):
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    raw_feature_spec.pop(_LABEL_KEY)
    
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()
    
    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    
    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)


def _build_estimator():
    feature_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_CATEGORICAL_FEATURE_KEYS)
    ]
    
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

    return classifier


def trainer_fn(hparams, schema):
    train_batch_size = 32
    eval_batch_size = 32
    
    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)
    
    train_input_fn = lambda: _input_fn(
        hparams.train_files,
        tf_transform_output,
        batch_size=train_batch_size)

    eval_input_fn = lambda: _input_fn(
        hparams.eval_files,
        tf_transform_output,
        batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps)

    serving_receiver_fn = lambda: _example_serving_receiver_fn(tf_transform_output, schema)

    exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='iris')

    estimator = _build_estimator()

    receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }
