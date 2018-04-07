from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from keras.applications.inception_v3 import decode_predictions
import base64
import json
import os
import numpy as np

PROJECTID = 'aerobic-coast-161423'
modelName = 'simple_mnist'
modelVersion = 'v4'

modelID = 'projects/{}/models/{}/versions/{}'.format(PROJECTID, modelName, modelVersion)

credentials = GoogleCredentials.get_application_default()
ml = discovery.build('ml', 'v1', credentials=credentials)

mapping = {
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


def transfer_decode_predictions(preds, top=None):
    preds = [(mapping[idx], pred) for idx, pred in enumerate(preds)]
    preds = sorted(preds, key=lambda pred: pred[1], reverse=True)
    if top:
        preds = preds[:top]
    return preds


with open('./data/fruits/apple/image_apple1.jpeg', 'rb') as f:
    b64_x = f.read()

b64_x = base64.urlsafe_b64encode(b64_x)
input_instance = dict(
    inputs=b64_x.decode('utf-8')
)
input_instance = json.loads(json.dumps(input_instance))

request_body = {"instances": [input_instance]}

request = ml.projects().predict(name=modelID, body=request_body)
try:
    response = request.execute()
    print(response)
    preds = response['predictions'][0]['outputs']
    preds = np.asarray(preds)
    preds = np.expand_dims(preds, 0)
    # print(preds)
    # print(len(preds[0]))

    print('Predicted:')
    for pred in transfer_decode_predictions(preds[0], top=5):
        print("Score {}, Label {}".format(pred[1], pred[0]))
    # for p in decode_predictions(preds, top=5)[0]:
    #     print("Score {}, Label {}".format(p[2], p[1]))

except errors.HttpError as err:
    print(err)
