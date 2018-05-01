import argparse

import tensorflow as tf
from keras import backend as keras_backend
from keras.applications import InceptionV3
from tensorflow.python.framework import graph_util


def load_inception():
    return InceptionV3(weights='imagenet')


def preprocess_b64_to_image():
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
    return g_input.as_graph_def()


def model_as_graph_no_variable(model):
    keras_session = keras_backend.get_session()
    inception_graph_def = keras_session.graph
    return graph_util.convert_variables_to_constants(
        keras_session,
        inception_graph_def.as_graph_def(),
        [model.output.name.replace(':0', '')]
    )


def export_keras_model_with_base_64_decode_input(model, path, preprocess_fn, fixed_model_graph):
    g_input_def = preprocess_fn()
    # inception_graph_def_with_no_variable = model_as_graph_no_variable(model)

    with tf.Graph().as_default() as g_combined:
        x = tf.placeholder(tf.string, name="input_b64")

        im, = tf.import_graph_def(g_input_def,
                                  input_map={'input:0': x},
                                  return_elements=["input_image:0"])
        pred, = tf.import_graph_def(fixed_model_graph,
                                    input_map={
                                        model.input.name: im,
                                        'batch_normalization_1/keras_learning_phase:0': False
                                    },
                                    return_elements=[model.output.name])

        with tf.Session() as sess:
            inputs = {"inputs": tf.saved_model.utils.build_tensor_info(x)}
            outputs = {"outputs": tf.saved_model.utils.build_tensor_info(pred)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
            sess.run(tf.global_variables_initializer())
            b = tf.saved_model.builder.SavedModelBuilder(path)
            b.add_meta_graph_and_variables(sess,
                                           [tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={'serving_default': signature})
            b.save()


def export_inception(path):
    model = load_inception()
    fixed_model_graph = model_as_graph_no_variable(model)
    export_keras_model_with_base_64_decode_input(model, path, preprocess_b64_to_image, fixed_model_graph)


parser = argparse.ArgumentParser(description='Export a model as protobuff file')
parser.add_argument("--model_dir", dest="model_dir", help="Path to store exported model",
                    metavar="FILE", default='./models/inception_v3')
args = parser.parse_args()

print('Exporting model to {}'.format(args.model_dir))
export_inception(args.model_dir)
print('Model exported')
