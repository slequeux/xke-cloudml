from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from tensorflow.python.framework import graph_util
import keras_gs
import os


def load_inception():
    print('Loading base model ...')
    model = InceptionV3(weights='imagenet')
    print('Loaded')
    return model


def export_inception_with_base64_decode(model, hparams):
    from keras.models import Model
    from keras.layers import Dense

    K.tensorflow_backend._get_available_gpus()

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

    #  flow
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(validation_split=0.25)

    train_generator = keras_gs.flow_from_google_storage(datagen,
        hparams.data_project, hparams.data_bucket, hparams.data_path,
        subset="training",
        target_size=(299, 299),
        batch_size=hparams.batch_size)

    validation_generator = keras_gs.flow_from_google_storage(datagen,
        hparams.data_project, hparams.data_bucket, hparams.data_path,
        subset="validation",
        target_size=(299, 299),
        batch_size=hparams.batch_size)

    history = transfer_model.fit_generator(
        train_generator,
        steps_per_epoch=hparams.steps_per_epoch,
        epochs=hparams.num_epochs,
        validation_data=validation_generator,
        validation_steps=hparams.validation_steps)

    acc = history.history['acc']
    loss = history.history['loss']
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
            b = tf.saved_model.builder.SavedModelBuilder(os.path.join(hparams.job_dir, "model"))
            b.add_meta_graph_and_variables(sess2,
                                           [tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={'serving_default': signature})
            b.save()
