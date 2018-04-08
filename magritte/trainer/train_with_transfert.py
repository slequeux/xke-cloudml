import keras
from keras import backend as K
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3 #, preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
# from os import listdir
# from os.path import isfile, join
# import datetime

BUCKET = 'sleq-ml-engine'


def load_inception():
    print('Loading base model ...')
    # https://medium.com/google-cloud/serverless-transfer-learning-with-cloud-ml-engine-and-keras-335435f31e15
    model = InceptionV3(weights='imagenet')
    # model = ''
    print('Loaded')
    return model


def load_dataset_from_gs(hparams):
    from keras.utils.np_utils import to_categorical
    from google.cloud import storage

    PROJECT = 'aerobic-coast-161423'
    BUCKET = 'sleq-ml-engine'
    PATH = 'data/raw/fruits/'

    category = 'fruits'
    storage_client = storage.Client(PROJECT)
    bucket = storage_client.get_bucket(BUCKET)
    labels_folder_iter = bucket.list_blobs(delimiter="/", prefix=PATH)
    list(labels_folder_iter) # populate blob_iter
    labels = [p[len(PATH):-1] for p in labels_folder_iter.prefixes]
    print('Found %s labels %s' % (len(labels), labels))
    X_dataset = np.zeros(shape=(1, 299, 299, 3))
    y_dataset = np.zeros(shape=1)
    for idx, label in enumerate(labels):
        print('Loading label %s' % label)
        dir_path = '%s%s/' % (PATH, label)
        img_blob_iter = bucket.list_blobs(delimiter="/", prefix=dir_path)
        paths = list(img_blob_iter)
        imgs = map(lambda blob: load_img_from_string(blob.download_as_string(storage_client), target_size=(299, 299)), paths)
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

    return X_dataset, y_dataset

def load_img_from_string(img_string, grayscale=False, target_size=None,
             interpolation='nearest'):
    from PIL import Image as pil_image
    import io
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(io.BytesIO(img_string))
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

# def load_dataset(hparams):
#     from keras.utils.np_utils import to_categorical
#
#     category = 'fruits'
#     print('Loading category %s' % category)
#     path = './data/%s' % category
#     labels = [f for f in listdir(path) if not(isfile(join(path, f)))]
#     #labels = ['apple', 'banana']
#     print('Found %s labels %s' % (len(labels), labels))
#     X_dataset = np.zeros(shape=(1, 299, 299, 3))
#     y_dataset = np.zeros(shape=1)
#     for idx, label in enumerate(labels):
#         print('Loading label %s' % label)
#         dir_path = './data/%s/%s' % (category, label)
#         paths = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
#         print('\tFound %i examples' % len(paths))
#         imgs = map(lambda path: image.load_img('%s/%s' % (dir_path, path), target_size=(299, 299)), paths)
#         imgs = map(lambda img: np.expand_dims(image.img_to_array(img), 0), imgs)
#         imgs = np.concatenate(imgs)
#         X_dataset = np.concatenate((X_dataset, imgs), axis=0)
#         labels = map(lambda path: idx, paths)
#         labels = np.array(labels)
#         y_dataset = np.concatenate((y_dataset, labels), axis=0)
#         print('\tLoaded')
#     print('Category %s loaded' % category)
#
#     X_dataset = X_dataset[1:, :, :, :]
#     y_dataset = y_dataset[1:]
#     y_dataset = to_categorical(y_dataset)
#
#     return X_dataset, y_dataset
#
#
def export_inception_with_base64_decode(model, hparams):
    from keras.models import Model
    from keras.layers import Dense
    from sklearn.model_selection import train_test_split

    print('Loading input dataset')
    X_dataset, y_dataset = load_dataset_from_gs(hparams)
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, test_size=0.25, random_state=42)
    num_classes = 9

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

    transfer_model.fit(X_train, y_train, epochs=hparams.num_epoch,
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
            # b = tf.saved_model.builder.SavedModelBuilder('./models/v2')
            b = tf.saved_model.builder.SavedModelBuilder(hparams.job_dir)
            b.add_meta_graph_and_variables(sess2,
                                           [tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={'serving_default': signature})
            b.save()
