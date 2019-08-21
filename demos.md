## Conference Demo Steps

#### Pre-demo Setup

* Clean up TFJobs, ACI, etc.
* Clean up storage account
* Scale up GPU nodes VMSS

#### Demos

1. GOT Web App

    * show web app 

2. Build / Train model

    * show Python training code
    * show TFJob yaml
    * deploy training ```kubectl apply -f ./k8s/tfjob-training.yaml```
    * show KF Dashboard
    * show logs
    * show Tensorboard 
    * show Azure Storage with output, models, etc.

3. Tensorflow Serving

    * talk about web app and need for serving program
    * show TF Serving yaml and k8s components
    * test API endpoint

        ```bash
        curl -X POST http://gotserving.domain.io:8501/v1/models/inception:predict -d "@./serving/daenerys-targaryen.json"

        python serving/inception_client.py --server gotserving.domain.io:8500 --image ./serving/jon-snow.jpg
        ```
    * show web app again 

4. Hyperparameter Optimization

    * talk about the need for hyperparameter tuning
    * show Helm chart
    * talk about the need for virtual node ```kubectl get node```
    * install TFJob chart
        
        ```bash
        helm install --name hyperparam ./hyperparameter/chart
        ```
    * show ACI's in portal
    * show Tensorboard 
    * show Katib in KF Dashboard 

5. Kubeflow Pipelines

    * show pipeline code
    * show KF Dashboard & Pipelines UI
    * create Run 
    * show where pipeline steps run as k8s pods
    * show Run output

6. Jupyter Notebooks

    * open Kubeflow dashboard 
    * browse to notebook and show GOT analysis
    * ```kubectl get pod -n kubeflow```




