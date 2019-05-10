FROM tensorflow/tensorflow:1.13.1
#FROM tensorflow/tensorflow:1.13.1-gpu

WORKDIR /

COPY ./images /images
COPY ./processimages.py /processimages.py

ENTRYPOINT [ "python","processimages.py" ]