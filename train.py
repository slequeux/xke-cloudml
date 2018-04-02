from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

BUCKET = 'sleq-ml-engine'
# https://medium.com/google-cloud/serverless-transfer-learning-with-cloud-ml-engine-and-keras-335435f31e15
model = InceptionV3(weights='imagenet')


def test_on_image(path='F:\\photos\\2011\\2011_07_conduite_ferrari\\IMG_2704.JPG'):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(
        img
    )
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # print(type(x))
    # print(x)
    preds = model.predict(x)

    print('Predicted:')
    for p in decode_predictions(preds, top=5)[0]:
        print("Score {}, Label {}".format(p[2], p[1]))


def export_inception_with_base64_decode():
    sess = K.get_session()
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

    # Convert to GraphDef
    g_input_def = g_input.as_graph_def()

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

        with tf.Session() as sess2:
            inputs = {"inputs": tf.saved_model.utils.build_tensor_info(x)}
            outputs = {"outputs": tf.saved_model.utils.build_tensor_info(pred)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )

            # save as SavedModel
            sess2.run(tf.global_variables_initializer())
            b = tf.saved_model.builder.SavedModelBuilder('./models/v2')
            # b = tf.saved_model.builder.SavedModelBuilder('gs://{}/simple-mnist/v2'.format(BUCKET))
            b.add_meta_graph_and_variables(sess2,
                                           [tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={'serving_default': signature})
            b.save()


# test_on_image()
export_inception_with_base64_decode()
