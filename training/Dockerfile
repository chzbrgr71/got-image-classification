#FROM tensorflow/tensorflow:1.13.1
FROM tensorflow/tensorflow:1.13.1-gpu

WORKDIR /

COPY ./images /images
COPY ./retrain.py /retrain.py

ENTRYPOINT [ "python","retrain.py" ]