from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from os import listdir
from os.path import isfile, join
import datetime

BUCKET = 'sleq-ml-engine'


def load_inception():
    print('Loading base model ...')
    # https://medium.com/google-cloud/serverless-transfer-learning-with-cloud-ml-engine-and-keras-335435f31e15
    model = InceptionV3(weights='imagenet')
    # model = ''
    print('Loaded')
    return model


def test_on_image(model, path='/Users/slequeux/Documents/test_images/20171226_175112.jpg'):
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


def load_dataset_2():
    """
    Just to understand how the default dataset works
    :return:
    """
    import requests
    from io import BytesIO
    from keras.utils import np_utils
    from keras.applications.inception_v3 import preprocess_input

    url = '/Users/slequeux/Downloads/tl_opera_capitol.npz'
    print('Loading data')
    # response = requests.get(url)
    with open(url, 'rb') as npzfile:
        dataset = np.load(BytesIO(npzfile.read()))

    X_dataset = dataset['features']
    y_dataset = dataset['labels']
    # print(type(X_dataset))
    # print(X_dataset.shape) # (200, 299, 299, 3)
    # print(type(y_dataset))
    # print(y_dataset.shape) # (200,)
    # print(X_dataset)
    # # [[[[  0.  62.  69.]
    # #    [  0.  64.  68.]
    # #    [  4.  74.  76.]
    # #    ...
    # print(y_dataset) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

    X_dataset = preprocess_input(X_dataset)
    y_dataset = np_utils.to_categorical(y_dataset)
    print(type(X_dataset))
    print(X_dataset.shape) # (200, 299, 299, 3)
    print(type(y_dataset))
    print(y_dataset.shape) # (200, 2)
    print(X_dataset)
    print(y_dataset)


def maybe_download(force=False):
    """
    Should get the training dataset from GCS if it is not already downloaded
    :return:
    """
    return None


def load_dataset(category='fruits'):
    from keras.utils.np_utils import to_categorical

    print('Loading category %s' % category)
    path = './data/%s' % category
    labels = [f for f in listdir(path) if not(isfile(join(path, f)))]
    #labels = ['apple', 'banana']
    print('Found %s labels %s' % (len(labels), labels))
    X_dataset = np.zeros(shape=(1, 299, 299, 3))
    y_dataset = np.zeros(shape=1)
    for idx, label in enumerate(labels):
        print('Loading label %s' % label)
        dir_path = './data/%s/%s' % (category, label)
        paths = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        print('\tFound %i examples' % len(paths))
        imgs = map(lambda path: image.load_img('%s/%s' % (dir_path, path), target_size=(299, 299)), paths)
        imgs = map(lambda img: np.expand_dims(image.img_to_array(img), 0), imgs)
        imgs = np.concatenate(imgs)
        X_dataset = np.concatenate((X_dataset, imgs), axis=0)
        labels = map(lambda path: idx, paths)
        labels = np.array(labels)
        y_dataset = np.concatenate((y_dataset, labels), axis=0)
        print('\tLoaded')
    print('Category %s loaded' % category)

    X_dataset = X_dataset[1:, :, :, :]
    y_dataset = y_dataset[1:]
    y_dataset = to_categorical(y_dataset)

    # print(y_dataset)
    # print(X_dataset.shape)
    # print(y_dataset.shape)
    return X_dataset, y_dataset


def export_inception_with_base64_decode(model):
    from keras.models import Model
    from keras.layers import Dense
    from sklearn.model_selection import train_test_split

    print('Loading input dataset')
    X_dataset, y_dataset = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, test_size=0.25, random_state=42)
    num_classes = y_train[0].size

    # Intermediate layer
    print('Defining the model')
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[311].output)
    x = intermediate_layer_model.output
    x = Dense(1024, activation='relu', name='dense_relu')(x)
    predictions = Dense(num_classes, activation='softmax', name='dense_softmax')(x)
    transfer_model = Model(inputs=intermediate_layer_model.input, outputs=predictions)
    for layer in transfer_model.layers:
        layer.trainable = False

    # Unfreeze the last layers, so that only these layers are trainable.
    transfer_model.layers[312].trainable = True
    transfer_model.layers[313].trainable = True

    print('Training ...')
    transfer_model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    transfer_model.fit(X_train, y_train, epochs=3,
                       verbose=2,
                       validation_data=(X_test, y_test))
    loss, acc = transfer_model.evaluate(X_test, y_test)
    print('Loss {}, Accuracy {}'.format(loss, acc))

    print('Exporting ...')
    sess = K.get_session()
    g_trans = sess.graph
    g_trans_def = graph_util.convert_variables_to_constants(sess,
                                                            g_trans.as_graph_def(),
                                                            [transfer_model.output.name.replace(':0', '')])

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
                                        transfer_model.input.name: im,
                                        'batch_normalization_1/keras_learning_phase:0': False
                                    },
                                    return_elements=[transfer_model.output.name])

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


start = datetime.datetime.now()

# test_on_image()
# export_inception_with_base64_decode()
model = load_inception()
export_inception_with_base64_decode(model)
# load_dataset_2()

end = datetime.datetime.now()
print('End : took %s' % (end - start))
