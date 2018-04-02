from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from keras.applications.inception_v3 import decode_predictions
import base64
import json
import os
import numpy as np


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    'C:\\Users\\Sylvain\\AppData\\Roaming\\gcloud\\application_default_credentials.json'

PROJECTID = 'aerobic-coast-161423'
modelName = 'simple_mnist'
modelVersion = 'v2'

modelID = 'projects/{}/models/{}/versions/{}'.format(PROJECTID, modelName, modelVersion)

credentials = GoogleCredentials.get_application_default()
ml = discovery.build('ml', 'v1', credentials=credentials)

with open('F:\\photos\\2011\\2011_07_conduite_ferrari\\IMG_2704.JPG', 'rb') as f:
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
    preds = response['predictions'][0]['outputs']
    preds = np.asarray(preds)
    preds = np.expand_dims(preds, 0)
    # print(preds)
    # print(len(preds))
    print('Predicted:')
    for p in decode_predictions(preds, top=5)[0]:
        print("Score {}, Label {}".format(p[2], p[1]))

except errors.HttpError as err:
    print(err)
