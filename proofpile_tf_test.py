"""proofpile_tf dataset."""

import tensorflow_datasets as tfds
from . import proofpile_tf


class ProofpileTfTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for proofpile_tf dataset."""
  DATASET_CLASS = proofpile_tf.ProofpileTf
  SPLITS = {
      'train': 779955,  # Number of fake train example
      'test': 13172,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
