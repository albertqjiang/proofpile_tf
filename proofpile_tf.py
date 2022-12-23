"""proofpile_tf dataset."""
import tensorflow_datasets as tfds
import json
import math

_DESCRIPTION = """
A TFDS version of the proof-pile dataset.
The proof-pile dataset can be found at [](https://huggingface.co/datasets/hoskinson-center/proof-pile).
"""

_CITATION = """Jiang2022proofpile_tf,
  author = {Jiang, Albert Q.},
  month = {12},
  title = {{proofpile_tfds: A tfds version of the proof-pile dataset.}},
  url = {https://github.com/albertqjiang/proofpile_tfds},
  version = {1.0.0},
  year = {2022}
"""

CONFIG_IDENTIFIER = "config"
SET_NAME_IDENTIFIER = "set_name"
SUBSET_NAME_IDENTIFIER = "subset_name"

ARXIV_STRING = "arxiv"
WIKI_STRING = "wiki"
STACK_EXCHANGE_STRING = "stack_exchange"
MATH_STRING = "MATH"
FILE_STRING = "file"
FORMAL_STRING = "formal"
BOOKS_STRING = "books"


class ProofpileTf(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for proofpile_tf dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'text': tfds.features.Text(),
            'context': tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://huggingface.co/datasets/hoskinson-center/proof-pile',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('gs://n2formal-public-data-europe/verifier/proofpile/pp_data.tar.gz')
    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'pp_data' / 'train_lines.jsonl'),
        "train_eval": self._generate_examples(path / 'pp_data' / 'train_eval_lines.jsonl'),
        'test': self._generate_examples(path / 'pp_data' / 'test_lines.jsonl'),
    }
  

  def _generate_examples(self, path):
    """Yields examples."""
    yield_register = set()
    with open(path) as fhand:
      for line in fhand.readlines():
        line_content = json.loads(line.strip())
        text = line_content['text']
        text_type = self.process_metadata(line_content['meta'])

        context = text_type
        chunks = math.ceil(len(text)/1500)

        for i in range(chunks):
          chunk = text[i*1500:(i+1)*1500]
          key = f"{text_type}-{hash(chunk)}"
          if key in yield_register:
            continue
          else:
            yield_register.add(key)
            
          datapoint = {
            "text": chunk,
            "context": context
          }
          context = chunk
          yield key, datapoint
  

  @staticmethod
  def process_metadata(metadata):
    def get_type(meta):
      if CONFIG_IDENTIFIER in meta:
          if meta[CONFIG_IDENTIFIER] == ARXIV_STRING:
              return ARXIV_STRING
          elif meta[CONFIG_IDENTIFIER] == WIKI_STRING:
              return WIKI_STRING
          else:
              print(meta[CONFIG_IDENTIFIER])
              raise AssertionError
      elif SET_NAME_IDENTIFIER in meta:
          if meta[SET_NAME_IDENTIFIER] == STACK_EXCHANGE_STRING:
              return STACK_EXCHANGE_STRING
          elif meta[SET_NAME_IDENTIFIER] == MATH_STRING:
              return MATH_STRING
          else:
              print(meta[SET_NAME_IDENTIFIER])
              raise AssertionError
      elif SUBSET_NAME_IDENTIFIER in meta:
          file_path = meta[FILE_STRING]
          file_type = file_path.split("/")
          if file_type[0] == FORMAL_STRING:
              file_type[-1] = file_type[-1].split(".")[0]
          elif file_type[0] == BOOKS_STRING:
              file_type[-1] = file_type[-1].rstrip(".tex")
          file_type = "/".join(file_type)
          return file_type
      else:
          return meta
    
    def sanitise_get_type(meta):
      supposed_type = get_type(meta)
      assert isinstance(supposed_type, str)
      return supposed_type.lower()

    return sanitise_get_type(metadata)
