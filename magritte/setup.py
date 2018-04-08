
'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='magritte',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Magritte from inception Keras model on Cloud ML Engine',
      author='Xebia',
      author_email='contact@xebia.fr',
      license='MIT',
      install_requires=[
          'keras', 'h5py', 'Pillow', 'google-cloud-storage'],
      zip_safe=False)