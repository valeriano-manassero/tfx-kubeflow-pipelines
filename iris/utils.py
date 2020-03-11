from typing import Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs


_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'species'


def _transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        transformed_features.pop(_transformed_name(_LABEL_KEY))
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: Text, tf_transform_output: tft.TFTransformOutput, batch_size: int = 200) -> tf.data.Dataset:
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY))
    return dataset


def _build_keras_model() -> tf.keras.Model:
    inputs = [keras.layers.Input(shape=(1,), name=_transformed_name(f)) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for _ in range(3):
        d = keras.layers.Dense(8, activation='relu')(d)
    output = keras.layers.Dense(3, activation='softmax')(d)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=absl.logging.info)
    return model


def _build_keras_model2() -> tf.keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=4, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.summary(print_fn=absl.logging.info)
    return model


def preprocessing_fn(inputs):
    outputs = {}

    for key in _FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

    return outputs


def run_fn(fn_args: TrainerFnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
