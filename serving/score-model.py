import os, sys
import tensorflow as tf

# pass in model path as arg (eg - /tf-output/latest_model)
# python score-model.py '../tf-output/latest_model'
model_path = sys.argv[1] 

label_lines = [line.rstrip() for line in tf.gfile.GFile(model_path + "/got_retrained_labels.txt")]

with tf.gfile.FastGFile(model_path + "/got_retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

total_score = 0

with tf.Session() as sess:
    images = ['jon-snow.jpg','night-king.jpg','cersei.jpg','robert-baratheon.jpg','theon-greyjoy.jpg','daenerys-targaryen.jpg','drogon.jpg','hodor.jpg','samwell.jpg','tyrion.jpg']

    for image in images:
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        score = predictions[0][top_k[0]]
        character = label_lines[top_k[0]]
        print(character + ': ' + str(score))
        total_score = total_score + score

avg_score = total_score / 10
print('---')
print('average model accuracy: ' + str(avg_score)) 