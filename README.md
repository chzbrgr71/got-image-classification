## Game of Thrones Image Classification Demo

This is demo code for my talk at [GlueCon 2019](http://gluecon.com).

"Machine Learning Made Easy on Kubernetes. DevOps for Data Scientists," May 23, 2019

### Game of Thrones Characters

From: https://gameofthrones.fandom.com 

Robert Baratheon (robert-baratheon)
Tyrion Lannister (tyrion-lannister)
Jon Snow (jon-snow)
Daenerys Targaryen (daenerys-targaryen)
Hodor (hodor)
Samwell Tarley (samwell-tarley)
Cersei Lannister (cersei-lannister)
Theon Greyjoy (theon-greyjoy)
Drogon (drogon)
Night King (night-king)


### Setup



```
docker run -it --rm --name got \
  --publish 6006:6006 \
  --publish 5000:5000 \
  --volume /Users/brianredmond/gopath/src/github.com/chzbrgr71/got-image-classification:/got-image-classification \
  --workdir /got-image-classification \
chzbrgr71/tf-workspace:2.0.0a0

python retrain-v2.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=/tmp/tensorflow/inception \
  --summaries_dir=/tmp/tensorflow/training_summaries/baseline \
  --output_graph=/got-image-classification/tf-output/retrained_graph.pb \
  --output_labels=/got-image-classification/tf-output/retrained_labels.txt \
  --image_dir=/got-image-classification/images \
  --saved_model_dir=/got-image-classification/tf-output/saved_models/1
  ```