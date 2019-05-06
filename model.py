import os, sys
import tensorflow as tf

with tf.gfile.FastGFile("./tf-output/latest_model/got_retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

print(graph_def)