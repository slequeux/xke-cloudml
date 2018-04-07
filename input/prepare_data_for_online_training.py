import os
import sys
from os import listdir
from os.path import isfile, join

import tensorflow as tf
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

FILE = os.path.realpath(__file__)
DEST = '%s/..' % os.path.dirname(FILE)


def load_feature_for_label(idx, label, category):
    print('Loading label %s' % label)
    dir_path = '%s/data/%s/%s' % (DEST, category, label)
    paths = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    print('\tFound %i examples' % len(paths))

    imgs = [image.load_img('%s/%s' % (dir_path, path), target_size=(299, 299)) for path in paths]
    imgs_arrays = [image.img_to_array(img) for img in imgs]

    return [to_feature(img_array, idx) for img_array in imgs_arrays]


def to_feature(data, label):
    image = tf.compat.as_bytes(data.tostring())
    return {
        'train/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'train/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }


def load_ds(category='fruits'):
    print('Loading category %s' % category)
    path = '%s/data/%s' % (DEST, category)
    labels = [f for f in listdir(path) if not(isfile(join(path, f)))]
    print('Found %s labels %s' % (len(labels), labels))
    list_of_list_of_features = [load_feature_for_label(idx, label, category) for idx, label in enumerate(labels)]
    return [item for sublist in list_of_list_of_features for item in sublist]


train_filename = '%s/data/tf_records/train.tfrecords' % DEST
eval_filename = '%s/data/tf_records/eval.tfrecords' % DEST

features = load_ds()
examples = [tf.train.Example(features=tf.train.Features(feature=feature)) for feature in features]

train_examples, eval_examples = train_test_split(examples, test_size=0.25, random_state=42)

writer = tf.python_io.TFRecordWriter(train_filename)
for example in train_examples:
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

writer = tf.python_io.TFRecordWriter(eval_filename)
for example in eval_examples:
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()
