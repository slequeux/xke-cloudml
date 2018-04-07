from google.cloud import storage
import os
import zipfile

PROJECT = 'aerobic-coast-161423'
BUCKET = 'sleq-ml-engine'
FILE = 'data.zip'
PATH = 'data/%s' % FILE
DEST = '%s/../' % os.path.realpath(__file__)


if os.path.exists('%s/%s' % (DEST, FILE)):
    print('Archive already downloaded, no need for doing it again')
else:
    storage_client = storage.Client(PROJECT)
    bucket = storage_client.get_bucket(BUCKET)
    blob = bucket.blob(PATH)
    print('Downloading %s/%s' % (BUCKET, PATH))
    blob.download_to_filename(DEST)
    print('Downloaded')


print('Unzip ...')
zip_ref = zipfile.ZipFile('%s/%s' % (DEST, FILE), 'r')
zip_ref.extractall(DEST)
zip_ref.close()
print('Files available in %s/data' % DEST)
