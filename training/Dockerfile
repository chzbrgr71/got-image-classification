FROM tensorflow/tensorflow:2.0.0a0

WORKDIR /

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY ./images /images
COPY ./retrain-keras.py /retrain-keras.py

ENTRYPOINT [ "python","retrain-keras.py" ]