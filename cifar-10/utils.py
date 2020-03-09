import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft

_IMAGE_KEY = 'image_raw'
_LABEL_KEY = 'label'

IMAGE_SIZE = 32


def _transformed_name(key):
    return key + '_xf'


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _image_parser(image_str):
    image = tf.image.decode_image(image_str, channels=3)
    image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.cast(image, tf.float32) / 255.
    return image


def _label_parser(label_id):
    label = tf.one_hot(label_id, 10)
    return label


def preprocessing_fn(inputs):
    outputs = {_transformed_name(_IMAGE_KEY): tf.compat.v2.map_fn(_image_parser, tf.squeeze(inputs[_IMAGE_KEY], axis=1),
                                                                  dtype=tf.float32),
               _transformed_name(_LABEL_KEY): tf.compat.v2.map_fn(_label_parser, tf.squeeze(inputs[_LABEL_KEY], axis=1),
                                                                  dtype=tf.float32)
               }
    return outputs


def _model_builder():
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name=_transformed_name(_IMAGE_KEY))
    d1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(inputs)
    d2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(d1)
    d3 = tf.keras.layers.Flatten()(d2)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(d3)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

    absl.logging.info(model.summary())
    return model


def _serving_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))

    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_labels = transformed_features.pop(_transformed_name(_LABEL_KEY))

    return tfma.export.EvalInputReceiver(features=transformed_features, labels=transformed_labels,
                                         receiver_tensors=serving_input_receiver.receiver_tensors)


def _input_fn(filenames, tf_transform_output, batch_size):
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(filenames, batch_size, transformed_feature_spec,
                                                                 reader=_gzip_reader_fn)

    return dataset.map(lambda features: (features, features.pop(_transformed_name(_LABEL_KEY))))


def trainer_fn(trainer_fn_args, schema):  # pylint: disable=unused-argument
    train_batch_size = 32
    eval_batch_size = 32

    tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

    train_input_fn = lambda: _input_fn(trainer_fn_args.train_files, tf_transform_output, batch_size=train_batch_size)

    eval_input_fn = lambda: _input_fn(trainer_fn_args.eval_files, tf_transform_output, batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=trainer_fn_args.train_steps)

    serving_receiver_fn = lambda: _serving_input_receiver_fn(tf_transform_output)

    exporter = tf.estimator.FinalExporter('cifar-10', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=trainer_fn_args.eval_steps, exporters=[exporter],
                                      name='cifar-10')

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=_model_builder(), config=run_config)

    eval_receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': eval_receiver_fn
    }
