import argparse

import tensorflow as tf
from keras import backend as keras_backend
from keras.applications import InceptionV3
from tensorflow.python.framework import graph_util


def from_keras():
    image = keras_backend.placeholder(shape=(1,), dtype=tf.string)

    def decode_b64_bytes(input_b64):
        print('input')
        print(input_b64)
        print('input2')
        print(input_b64[:])
        # input_b64_str = tf.cast(input_b64, dtype=tf.string)
        # print(input_b64_str)
        input_bytes = tf.decode_base64(input_b64[:][0])
        print('bytes')
        print(input_bytes)
        # return input_bytes
        img = tf.image.decode_image(input_bytes[0])
        return img
        # return tf.image.convert_image_dtype(img, dtype=tf.float32)

    # inception = InceptionV3(weights='imagenet')
    # inception.summary()

    # input_layer = models.Input(shape=(1,), dtype=tf.string)
    # output = layers.Lambda(function=decode_b64_bytes, name='Base64Decode')(input_layer)
    #
    # convertion_model = models.Model(inputs=[input_layer], outputs=output)

    # convertion_model.summary()

    # model = models.Sequential()
    # model.add(layers.Input(shape=(1,), dtype=tf.string))
    # model.add(layers.Lambda(decode_b64_bytes, input_shape=(1,), name='Base64Decode'))
    # model.add(inception)
    # model.add(layers.Input(shape=(1,), dtype=tf.string, tensor=image))
    # model.add(layers.Lambda(lambda b64: tf.decode_base64(b64), input_shape=(1,)))
    # model.add(layers.Lambda(lambda input_bytes: tf.image.decode_image(input_bytes)))
    # model.add(layers.Lambda(lambda img: tf.image.convert_image_dtype(img, dtype=tf.float32)))
    # model.add(layers.Lambda(lambda img: tf.expand_dims(img, axis=0)))

    # model.summary()


def load_inception():
    return InceptionV3(weights='imagenet')


def export_keras_model_with_base_64_decode_input(model, path):
    keras_session = keras_backend.get_session()
    inception_graph_def = keras_session.graph
    inception_graph_def_with_no_variable = graph_util.convert_variables_to_constants(
        keras_session,
        inception_graph_def.as_graph_def(),
        [model.output.name.replace(':0', '')]
    )

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
        pred, = tf.import_graph_def(inception_graph_def_with_no_variable,
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
    export_keras_model_with_base_64_decode_input(model, path)


parser = argparse.ArgumentParser(description='Export a model as protobuff file')
parser.add_argument("--model_dir", dest="model_dir", help="Path to store exported model",
                    metavar="FILE", default='./models/inception_v3')
args = parser.parse_args()

print('Exporting model to {}'.format(args.model_dir))
export_inception(args.model_dir)
print('Model exported')
