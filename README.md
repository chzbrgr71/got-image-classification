## Game of Thrones Image Classification Demo

This is demo code for my talk at [GlueCon 2019](http://gluecon.com).

"Machine Learning Made Easy on Kubernetes. DevOps for Data Scientists," May 23, 2019

### Game of Thrones Characters

From: https://gameofthrones.fandom.com 

* Robert Baratheon (robert-baratheon)
* Tyrion Lannister (tyrion-lannister)
* Jon Snow (jon-snow)
* Daenerys Targaryen (daenerys-targaryen)
* Hodor (hodor)
* Samwell Tarley (samwell-tarley)
* Cersei Lannister (cersei-lannister)
* Theon Greyjoy (theon-greyjoy)
* Drogon (drogon)
* Night King (night-king)

https://www.tensorflow.org/hub/tutorials/image_retraining

### Local testing/training

* Testing in local Docker container interactively

  ```bash
  docker run -it --rm --name got \
    --publish 6006:6006 \
    --publish 5000:5000 \
    --volume /Users/brianredmond/gopath/src/github.com/chzbrgr71/got-image-classification:/got-image-classification \
    --workdir /got-image-classification \
  tensorflow/tensorflow:1.13.1

  python ./training/retrain.py \
    --bottleneck_dir=/tmp/tensorflow/bottlenecks \
    --model_dir=/tmp/tensorflow/inception \
    --summaries_dir=/got-image-classification/tf-output \
    --output_graph=/got-image-classification/tf-output \
    --output_labels=/got-image-classification/tf-output \
    --image_dir=/got-image-classification/training/images \
    --saved_model_dir=/got-image-classification/tf-output \
    --how_many_training_steps 50

  tensorboard --logdir=/got-image-classification/tf-output/training_summaries
  ```

* Create container image

  ```bash
  # set image tag depending on target cpu/gpu
  export IMAGE_TAG=1.141
  export IMAGE_TAG=1.0-gpu
  export ACRNAME=briaracr

  # build/push (ACR or Docker)
  az acr build -t chzbrgr71/got-image-training:$IMAGE_TAG -r $ACRNAME ./training

  docker build -t chzbrgr71/got-image-training:$IMAGE_TAG -f ./training/Dockerfile ./training
  docker push chzbrgr71/got-image-training:$IMAGE_TAG
  ```

* Run local

  ```bash
  docker run -d --name train --volume /Users/brianredmond/gopath/src/github.com/chzbrgr71/got-image-classification:/got-image-classification chzbrgr71/got-image-training:$IMAGE_TAG "--bottleneck_dir=/tmp/tensorflow/bottlenecks" "--model_dir=/tmp/tensorflow/inception" "--summaries_dir=/got-image-classification/tf-output/training_summaries/baseline" "--output_graph=/got-image-classification/tf-output/retrained_graph.pb" "--output_labels=/got-image-classification/tf-output/retrained_labels.txt" "--image_dir=/images" "--saved_model_dir=/got-image-classification/tf-output/saved_models/1"
  ```

* Tensorboard local
  ```bash
  export IMAGE_TAG=1.1
  export ACRNAME=briaracr

  # build/push (ACR or Docker)
  az acr build -t chzbrgr71/tensorboard:$IMAGE_TAG -r $ACRNAME -f ./tensorboard/Dockerfile ./tensorboard

  docker build -t chzbrgr71/tensorboard:$IMAGE_TAG -f ./tensorboard/Dockerfile ./tensorboard
  docker push chzbrgr71/tensorboard:$IMAGE_TAG

  # run
  docker run -d --name tb -p 6006:6006 --volume /Users/brianredmond/gopath/src/github.com/chzbrgr71/got-image-classification/tf-output:/tf-output chzbrgr71/tensorboard:$IMAGE_TAG "--logdir" "/tf-output/training_summaries"
    ```

### Kubernetes Setup

* Create Azure Kubernetes Service

* Storage (Azure Files Static)

    Azure Files Docs: https://docs.microsoft.com/en-us/azure/aks/azure-files-volume 

    ```bash
    export AKS_PERS_STORAGE_ACCOUNT_NAME=briar$RANDOM
    export AKS_PERS_RESOURCE_GROUP=briar-aks-ml-200
    export AKS_PERS_LOCATION=eastus
    export AKS_PERS_SHARE_NAME=aksshare

    # Create the storage account
    az storage account create -n $AKS_PERS_STORAGE_ACCOUNT_NAME -g $AKS_PERS_RESOURCE_GROUP -l $AKS_PERS_LOCATION --sku Standard_LRS

    # Export the connection string as an environment variable, this is used when creating the Azure file share
    export AZURE_STORAGE_CONNECTION_STRING=`az storage account show-connection-string -n $AKS_PERS_STORAGE_ACCOUNT_NAME -g $AKS_PERS_RESOURCE_GROUP -o tsv`

    # Create the file share
    az storage share create -n $AKS_PERS_SHARE_NAME

    # Get storage account key
    STORAGE_KEY=$(az storage account keys list --resource-group $AKS_PERS_RESOURCE_GROUP --account-name $AKS_PERS_STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)

    # Echo storage account name and key
    echo Storage account name: $AKS_PERS_STORAGE_ACCOUNT_NAME
    echo Storage account key: $STORAGE_KEY

    kubectl create secret generic azure-file-secret --from-literal=azurestorageaccountname=$AKS_PERS_STORAGE_ACCOUNT_NAME --from-literal=azurestorageaccountkey=$STORAGE_KEY
    ```

### Training in Kubernetes

* Kubernetes job

  ```bash
  kubectl apply -f ./k8s/training-job.yaml

  kubectl apply -f ./k8s/tensorboard.yaml
  ```

### Inference

* Local python script

  ```bash
  # testing
  python ./serving/label-image.py ./serving/hodor.jpg

  hodor (score = 0.98614)
  samwell tarley (score = 0.01098)
  tyrion lannister (score = 0.00151)
  theon greyjoy (score = 0.00105)
  jon snow (score = 0.00020)
  robert baratheon (score = 0.00008)
  drogon (score = 0.00002)
  night king (score = 0.00001)
  daenerys targaryen (score = 0.00000)
  cersei lannister (score = 0.00000)
  ```

* TF Serving

  ```bash
  docker run -d --rm --name serving_base tensorflow/serving:1.13.0
  docker cp ./tf-output/saved_models serving_base:/models/inception
  docker commit --change "ENV MODEL_NAME inception" serving_base chzbrgr71/got_serving:1.0
  docker kill serving_base
  docker run -p 8500:8500 --name serving -t chzbrgr71/got_serving:1.0 &

  python serving/inception_client.py --server localhost:8500 --image ./serving/hodor.jpg
  python serving/inception_client.py --server localhost:8500 --image ./serving/tyrion.jpg
  python serving/inception_client.py --server localhost:8500 --image ./serving/night-king.jpg
  ```

  ```bash
  docker run -d --name serving \
    --publish 8500:8500 \
    --volume /Users/brianredmond/gopath/src/github.com/chzbrgr71/got-image-classification/tf-output/saved_models:/models/inception \
    --env MODEL_NAME=inception \
  tensorflow/serving:1.13.0
  ```

  ```bash
  kubectl apply -f ./k8s/serving.yaml

  python serving/inception_client.py --server 23.96.109.181:8500 --image ./serving/night-king.jpg
  ```

### Export model

https://www.tensorflow.org/js/tutorials/conversion/import_keras#alternative_use_the_python_api_to_export_directly_to_tfjs_layers_format 