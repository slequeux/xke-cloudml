import argparse

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image

from utils import is_valid_file


def load_inception():
    return InceptionV3(weights='imagenet')


def test_on_image(model, image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    print('Predicted:')
    for p in decode_predictions(preds, top=5)[0]:
        print("Score {}, Label {}".format(p[2], p[1]))


parser = argparse.ArgumentParser(description='Perform locally a prediction on InceptionV3')
parser.add_argument("--image", dest="image_path", required=True, help="File to perform the prediction on",
                    metavar="FILE", type=lambda x: is_valid_file(parser, x))
parser.add_argument("--model_path", dest="model_path", required=False,
                    help="Path to load a model in protobuff format (if not provided, will use default InceptionV3)")
args = parser.parse_args()


if args.model_path:
    raise NotImplementedError('Custom model not supported')
else:
    model = load_inception()

test_on_image(model, args.image_path)
