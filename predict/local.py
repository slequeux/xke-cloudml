from __future__ import absolute_import

import argparse
import base64

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.contrib import predictor


fruit_mapping = {
    0: 'apple',
    1: 'banana',
    2: 'grape',
    3: 'kiwi',
    4: 'mango',
    5: 'orange',
    6: 'pineapple',
    7: 'raspberry',
    8: 'strawberry'
}


def predict_from_saved_model(model_path, image_path, decode_fn, top=5):
    predict_fn = predictor.from_saved_model(export_dir=model_path)

    with open(image_path, 'rb') as f:
        b64_x = f.read()
    b64_x = base64.urlsafe_b64encode(b64_x)
    input_instance = {
        'inputs': [b64_x]
    }

    preds = predict_fn(input_instance)['outputs'][0]
    if decode_fn == 'inception':
        preds = np.expand_dims(preds, 0)
        print('Predicted:')
        for p in decode_predictions(preds, top=5)[0]:
            print("Score {}, Label {}".format(p[2], p[1]))
    elif decode_fn == 'transfert':
        preds = [(fruit_mapping[idx], pred) for idx, pred in enumerate(preds)]
        preds = sorted(preds, key=lambda pred: pred[1], reverse=True)
        if top:
            preds = preds[:top]
        print('Predicted:')
        for pred in preds:
            print("Score {}, Label {}".format(pred[1], pred[0]))
    else:
        print('Predicted scores:')
        print(preds)


def predict_from_inception(image_path, top=5):
    model = InceptionV3(weights='imagenet')

    image_buffer = open(image_path, 'r')
    img = image.load_img(image_buffer, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    print('Predicted:')
    for p in decode_predictions(preds, top=top)[0]:
        print("Score {}, Label {}".format(p[2], p[1]))


parser = argparse.ArgumentParser(description='Perform locally a prediction on InceptionV3')
parser.add_argument("--image", dest="image_path", required=True, help="File to perform the prediction on",
                    metavar="FILE")
parser.add_argument("--model_path", dest="model_path", required=False,
                    help="Path to load a model in protobuff format (if not provided, will use default InceptionV3)")
parser.add_argument("--decode_preds_fn", dest="decode_fn", required=False,
                    help="Function to decode prediction (supported values are inception and transfert)")
parser.add_argument("--top", dest="top", required=False, default=5, help="Top N predictions to display", metavar='N')
args = parser.parse_args()


if args.model_path:
    predict_from_saved_model(args.model_path, args.image_path, args.decode_fn, args.top)
else:
    predict_from_inception(args.image_path, int(args.top))
