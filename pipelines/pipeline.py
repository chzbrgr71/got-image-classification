import kfp.dsl as dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name='GameOfThrones',
    description='Game of Thrones Tensorflow image classification demo'
)
def got_image_pipeline(
    trainingsteps=4000,
    learningrate=0.01,
    trainbatchsize=100,
):

    persistent_volume_name = 'azure-files'
    persistent_volume_path = '/tf-output'
    azure_file_secret_name = 'azure-file-secret'
    azure_file_share_name = 'pipelines'
    field_path = 'metadata.name'

    operations = {}

    # preprocess images
    operations['preprocess'] = dsl.ContainerOp(
        name='preprocess',
        image='gotcr.azurecr.io/chzbrgr71/got-image-preprocess:2.00',
        arguments=[
            '--bottleneck_dir', "/tf-output/bottlenecks",
            '--image_dir', '/images'
        ]
    )

    # train
    operations['train'] = dsl.ContainerOp(
        name='train',
        image='gotcr.azurecr.io/chzbrgr71/got-image-training:2.00',
        arguments=[
            '--bottleneck_dir', "/tmp/tensorflow/bottlenecks",
            '--model_dir', "/tmp/tensorflow/inception",
            '--summaries_dir', '/tf-output',
            '--output_graph', '/tf-output',
            '--output_labels', '/tf-output',
            '--image_dir', "/images",
            '--saved_model_dir', '/tf-output',
            '--how_many_training_steps', trainingsteps,
            '--learning_rate', learningrate,
            '--train_batch_size', trainbatchsize
        ]
    )
    operations['train'].after(operations['preprocess'])

    # score model
    operations['score'] = dsl.ContainerOp(
        name='score',
        image='gotcr.azurecr.io/chzbrgr71/got-model-scoring:2.01',
        arguments=[
            '/tf-output/latest_model'
        ]
    )
    operations['score'].after(operations['train'])

    # convert onnx
    operations['onnx'] = dsl.ContainerOp(
        name='onnx',
        image='gotcr.azurecr.io/chzbrgr71/onnx-convert:2.00',
        arguments=[
            'show',
            '--dir', '/tf-output/latest_model/exported_model/1/',
            '--tag_set', 'serve',
            '--signature_def', 'serving_default'
        ]
    )
    operations['onnx'].after(operations['score'])

    # convert tflite
    operations['convert-tflite'] = dsl.ContainerOp(
        name='convert-tflite',
        image='gotcr.azurecr.io/chzbrgr71/tflite-convert:2.00',
        arguments=[
            '--graph_def_file', '/tf-output/latest_model/got_retrained_graph.pb',
            '--output_file', '/tf-output/latest_model/optimized_graph.lite',
            '--input_format', 'TENSORFLOW_GRAPHDEF',
            '--output_format', 'TFLITE',
            '--input_sedhape', '1,299,299,3',
            '--input_array', 'Mul',
            '--output_array', 'final_result',
            '--inference_type', 'FLOAT',
            '--input_data_type', 'FLOAT'
        ]
    )    
    operations['convert-tflite'].after(operations['score'])

    # copy models to external storage
    operations['export-to-cloud'] = dsl.ContainerOp(
        name='export-to-cloud',
        image='alpine',
        command=['cp'],
        arguments=[
            '/tf-output/latest_model/got_retrained_graph.pb', 
            '/tf-output/latest_model/got_retrained_graph-latest.pb'
        ]
    )
    operations['export-to-cloud'].after(operations['onnx']).after(operations['convert-tflite'])

    for _, op in operations.items():
        op.add_volume(
                k8s_client.V1Volume(
                    azure_file=k8s_client.V1AzureFileVolumeSource(
                        secret_name=azure_file_secret_name,
                        share_name=azure_file_share_name,
                        read_only=False),
                        name=persistent_volume_name)
                ) \
            .add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=persistent_volume_path, 
                name=persistent_volume_name)
                ) \
            .add_env_variable(k8s_client.V1EnvVar(name='MSG', value='HELLO!')
                ) \
            .add_env_variable(k8s_client.V1EnvVar(name='KUBE_POD_NAME',
                value_from=k8s_client.V1EnvVarSource(
                    field_ref=k8s_client.V1ObjectFieldSelector(field_path=field_path)
            )))

if __name__ == '__main__':
   import kfp.compiler as compiler
   compiler.Compiler().compile(got_image_pipeline, __file__ + '.tar.gz')