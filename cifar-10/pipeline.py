import os
from typing import Text

from kfp import onprem

from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input


_pipeline_name = 'cifar-10'

_persistent_volume_claim = 'tfx-pvc'
_persistent_volume = 'tfx-pv'
_persistent_volume_mount = '/mnt'

_input_base = os.path.join(_persistent_volume_mount, 'cifar-10')
_output_base = os.path.join(_persistent_volume_mount, 'pipelines')
_tfx_root = os.path.join(_output_base, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

_data_root = os.path.join(_input_base, 'data')

_module_file = os.path.join(_input_base, 'utils.py')

_serving_model_dir = os.path.join(_output_base, _pipeline_name, 'serving_model')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text, module_file: Text,
                     serving_model_dir: Text, direct_num_workers: int) -> pipeline.Pipeline:
    examples = external_input(data_root)
    input_split = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train.tfrecord'),
        example_gen_pb2.Input.Split(name='eval', pattern='test.tfrecord')
    ])
    example_gen = ImportExampleGen(input_base=examples, input_config=input_split)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    validate_stats = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                      schema=infer_schema.outputs['schema'])
    transform = Transform(examples=example_gen.outputs['examples'], schema=infer_schema.outputs['schema'],
                          module_file=module_file)

    trainer = Trainer(module_file=module_file, transformed_examples=transform.outputs['transformed_examples'],
                      schema=infer_schema.outputs['schema'], transform_graph=transform.outputs['transform_graph'],
                      train_args=trainer_pb2.TrainArgs(num_steps=10000), eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )

    model_analyzer = Evaluator(examples=example_gen.outputs['examples'], model_exports=trainer.outputs['model'],
                               feature_slicing_spec = evaluator_pb2.FeatureSlicingSpec(specs=[
                                   evaluator_pb2.SingleSlicingSpec()
                               ]))

    model_validator = ModelValidator(examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

    pusher = Pusher(model=trainer.outputs['model'], model_blessing=model_validator.outputs['blessing'],
                    push_destination=pusher_pb2.PushDestination(
                        filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[example_gen, statistics_gen, infer_schema, validate_stats, transform, trainer, model_analyzer,
                    model_validator, pusher],
        beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers]
    )


if __name__ == '__main__':
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(kubeflow_metadata_config=metadata_config,
                                                                tfx_image=tfx_image,
                                                                pipeline_operator_funcs=([
                                                                    onprem.mount_pvc(_persistent_volume_claim,
                                                                                     _persistent_volume,
                                                                                     _persistent_volume_mount),
                                                                    kubeflow_dag_runner._mount_config_map_op('metadata-db-configmap'),
                                                                    kubeflow_dag_runner._mount_secret_op('metadata-db-secrets')
                                                                ]
                                                                ))

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(_create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        direct_num_workers=0))
