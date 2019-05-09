import kfp.dsl as dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name='DogsVCats',
    description='Simple TF CNN for binary classifier between dogs and cats'
)
def dogsandcats_train(
    base_path='/mnt/azure',
    epochs=5,
    batch=32,
    learning_rate=0.0001
):

    persistent_volume_name = 'azure'
    persistent_volume_path = '/mnt/azure/'

    operations = {}

    # preprocess data
    operations['preprocess'] = dsl.ContainerOp(
        name='preprocess',
        image='tlaloc.azurecr.io/kubeflow/preprocess',
        command=['python'],
        arguments=[
            '/scripts/data.py',
            '--base_path', base_path,
            '--data', 'data/PetImages',
            '--target', 'dataset.txt',
            '--img_size', '160'
        ]
    )

    # train
    operations['train'] = dsl.ContainerOp(
        name='train',
        image='tlaloc.azurecr.io/kubeflow/train',
        command=['python'],
        arguments=[
            '/scripts/train.py',
            '--base_path', base_path,
            '--data', 'data', 
            '--epochs', epochs, 
            '--batch', batch, 
            '--image_size', '160', 
            '--lr', learning_rate, 
            '--outputs', 'model', 
            '--dataset', 'dataset.txt'
        ]
    )
    operations['train'].after(operations['preprocess'])

    # score
    operations['score'] = dsl.ContainerOp(
        name='score',
        image='tlaloc.azurecr.io/kubeflow/score',
        command=['python'],
        arguments=[
            '/scripts/score.py',
            '--base_path', base_path,
            '--model', 'model/latest.h5'
        ]
    )
    operations['score'].after(operations['train'])

    # #release
    # operations['release'] = dsl.ContainerOp(
    #     name='release',
    #     image='tlaloc.azurecr.io/kubeflow/release'
    # )
    # operations['release'].after(operations['score'])

    for _, op in operations.items():
        op.add_volume(
                k8s_client.V1Volume(
                    host_path=k8s_client.V1HostPathVolumeSource(
                        path=persistent_volume_path),
                        name=persistent_volume_name)
                ) \
            .add_volume_mount(k8s_client.V1VolumeMount(
                mount_path=persistent_volume_path, 
                name=persistent_volume_name))

if __name__ == '__main__':
   import kfp.compiler as compiler
   compiler.Compiler().compile(dogsandcats_train, __file__ + '.tar.gz')