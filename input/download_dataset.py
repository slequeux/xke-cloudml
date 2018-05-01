from google.cloud import storage
import os
import zipfile

PROJECT = 'aerobic-coast-161423'
BUCKET = 'sleq-ml-engine'
FILE = 'data.zip'
PATH = 'data/%s' % FILE
DEST = '%s/../data' % os.path.dirname(os.path.realpath(__file__))

ARCHIVE_FILE = '%s/%s' % (DEST, FILE)

if os.path.exists(ARCHIVE_FILE):
    print('Archive already downloaded, no need for doing it again')
else:
    storage_client = storage.Client(PROJECT)
    bucket = storage_client.get_bucket(BUCKET)
    blob = bucket.blob(PATH)
    print('Downloading %s/%s to %s' % (BUCKET, PATH, ARCHIVE_FILE))
    blob.download_to_filename(ARCHIVE_FILE)
    print('Downloaded')


print('Unzip ...')
zip_ref = zipfile.ZipFile(ARCHIVE_FILE, 'r')
zip_ref.extractall(DEST)
zip_ref.close()
print('Files available in %s/data' % DEST)
