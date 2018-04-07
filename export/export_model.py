import argparse

import tensorflow as tf
from keras import backend as keras_backend
from tensorflow.python.framework import graph_util

from predict.local import load_inception


def add_base_64_decode_input_layers(model):
    """
    Use a model in input and add some layers so input could be base 64 encoded images.
    :param model: model to convert, must take images (float64 arrays) as input
    :return: tupple of tensors representing input and output tensors
    """
    sess = keras_backend.get_session()
    g_trans = sess.graph
    g_trans_def = graph_util.convert_variables_to_constants(sess,
                                                            g_trans.as_graph_def(),
                                                            [model.output.name.replace(':0', '')])

    # Step 1 : Build a graph that converts image
    with tf.Graph().as_default() as g_input:
        input_b64 = tf.placeholder(
            shape=(1,),
            dtype=tf.string,
            name='input')
        input_bytes = tf.decode_base64(input_b64[0])
        img = tf.image.decode_image(input_bytes)
        image_f = tf.image.convert_image_dtype(img, dtype=tf.float32)
        input_image = tf.expand_dims(image_f, axis=0)
        tf.identity(input_image, name='input_image')
    g_input_def = g_input.as_graph_def()

    # Step 2 : create a graph that chain image conversion graph and model execution
    with tf.Graph().as_default() as g_combined:
        x = tf.placeholder(tf.string, name="input_b64")

        im, = tf.import_graph_def(g_input_def,
                                  input_map={'input:0': x},
                                  return_elements=["input_image:0"])
        pred, = tf.import_graph_def(g_trans_def,
                                    input_map={
                                        model.input.name: im,
                                        'batch_normalization_1/keras_learning_phase:0': False
                                    },
                                    return_elements=[model.output.name])

    return x, pred


def export_current_graph_for_serving(input_tensor, output_tensor, path):
    with tf.Session() as sess:
        inputs = {"inputs": tf.saved_model.utils.build_tensor_info(input_tensor)}
        outputs = {"outputs": tf.saved_model.utils.build_tensor_info(output_tensor)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        b = tf.saved_model.builder.SavedModelBuilder(path)
        b.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       signature_def_map={'serving_default': signature})
        b.save()


def export_inception(path):
    model = load_inception()
    input_tensor, output_tensor = add_base_64_decode_input_layers(model)
    export_current_graph_for_serving(input_tensor, output_tensor, path)


parser = argparse.ArgumentParser(description='Export a model as protobuff file')
parser.add_argument("--dest", dest="model_dir", help="Path to store exported model",
                    metavar="FILE", default='./models/inception_v3')
args = parser.parse_args()

export_inception(args.model_dir)
