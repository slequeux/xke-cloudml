import argparse
import base64
import json

import numpy as np
from googleapiclient import discovery
from googleapiclient import errors
from keras.applications.inception_v3 import decode_predictions
from oauth2client.client import GoogleCredentials

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


def predict(project_id, model_name, model_version, image_path, decode_fn, top=5):
    model_id = 'projects/{}/models/{}/versions/{}'.format(project_id, model_name, model_version)
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials)

    with open(image_path, 'rb') as f:
        b64_x = f.read()
    b64_x = base64.urlsafe_b64encode(b64_x)
    input_instance = {'inputs': b64_x.decode('utf-8')}
    input_instance = json.loads(json.dumps(input_instance))
    request_body = {"instances": [input_instance]}
    request = ml.projects().predict(name=model_id, body=request_body)

    try:
        response = request.execute()
        preds = response['predictions'][0]['outputs']
        preds = np.asarray(preds)
        preds = np.expand_dims(preds, 0)

        if decode_fn == 'inception':
            print('Predicted:')
            for p in decode_predictions(preds, top=top)[0]:
                print("Score {}, Label {}".format(p[2], p[1]))
        elif decode_fn == 'transfert':
            preds = [(fruit_mapping[idx], pred) for idx, pred in enumerate(preds[0])]
            preds = sorted(preds, key=lambda pred: pred[1], reverse=True)
            if top:
                preds = preds[:top]
            print('Predicted:')
            for pred in preds:
                print("Score {}, Label {}".format(pred[1], pred[0]))
        else:
            print('Predicted scores:')
            print(preds)

    except errors.HttpError as err:
        print(err)

    pass


parser = argparse.ArgumentParser(description='Perform locally a prediction on InceptionV3')
parser.add_argument("--image", dest="image_path", required=True, help="File to perform the prediction on",
                    metavar="FILE")
parser.add_argument("--project_id", dest="project_id", required=True, help="GCP Project")
parser.add_argument("--model_name", dest="model_name", required=True, help="CloudML model name")
parser.add_argument("--model_version", dest="model_version", required=True, help="CloudML model version")
parser.add_argument("--decode_preds_fn", dest="decode_fn", required=False,
                    help="Function to decode prediction (supported values are inception and transfert)")
parser.add_argument("--top", dest="top", required=False, default=5, help="Top N predictions to display", metavar='N')
args = parser.parse_args()

predict(args.project_id, args.model_name, args.model_version, args.image_path, args.decode_fn, int(args.top))
