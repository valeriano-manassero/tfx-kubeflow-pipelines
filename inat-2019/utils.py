import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

_IMAGE_KEY = 'image_raw'
_LABEL_KEY = 'label'

IMAGE_SIZE = 224


def _transformed_name(key):
    if key == _IMAGE_KEY:
        return 'image_raw_xf_input'
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')


def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _fill_in_missing(x):
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def _image_parser(image_str):
    image = tf.image.decode_image(image_str)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.cast(image, tf.float32) / 127.5 - 1.
    return image


def _label_parser(label_id):
    label = tf.one_hot(label_id, 10)
    return label


def preprocessing_fn(inputs):
    outputs = {_transformed_name(_IMAGE_KEY): tf.compat.v2.map_fn(lambda x: _image_parser(x),
                                                                  _fill_in_missing(inputs[_IMAGE_KEY]),
                                                                  dtype=tf.float32),
               _transformed_name(_LABEL_KEY): tf.compat.v2.map_fn(lambda x: _label_parser(x),
                                                                  _fill_in_missing(inputs[_LABEL_KEY]),
                                                                  dtype=tf.float32)}
    return outputs


def _input_fn(filenames, tf_transform_output, batch_size):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

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
    labels = transformed_features.pop(_transformed_name(_LABEL_KEY))

    receiver_tensors = {'examples': serialized_tf_example}

    return tfma.export.EvalInputReceiver(
        features=transformed_features,
        receiver_tensors=receiver_tensors,
        labels=labels)


def _example_serving_receiver_fn(tf_transform_output, schema):
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))

    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)


def _build_estimator():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(682, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metric='accuracy')

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

    return estimator


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

    exporter = tf.estimator.FinalExporter('inat-2019', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='inat-2019')

    estimator = _build_estimator()

    receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }
