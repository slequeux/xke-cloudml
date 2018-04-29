import argparse
import os

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
import trainer.train_with_transfert as trainer

def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""
    model = trainer.load_inception()
    trainer.export_inception_with_base64_decode(model, hparams)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.\
      """,
        type=int,
        default=3
    )

    parser.add_argument(
        '--steps-per-epoch',
        help="""\
      Number of steps per epoch\
      """,
        type=int,
        default=3
    )

    parser.add_argument(
        '--validation-steps',
        help="""\
      Number of validation steps.\
      """,
        type=int,
        default=3
    )

    parser.add_argument(
        '--batch-size',
        help="""\
      Size of batch\
      """,
        type=int,
        default=32
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)
