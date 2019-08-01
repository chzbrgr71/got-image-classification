#FROM tensorflow/serving:1.13.0
FROM tensorflow/tensorflow:1.13.1

WORKDIR /

COPY ./*.jpg /
COPY ./score-model.py /score-model.py

ENTRYPOINT [ "python","score-model.py" ]