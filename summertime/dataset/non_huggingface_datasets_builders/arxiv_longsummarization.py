import os
import json
import datasets


"""Arxiv dataset."""


_CITATION = """
@article{Cohan_2018,
   title={A Discourse-Aware Attention Model for Abstractive Summarization of
            Long Documents},
   url={http://dx.doi.org/10.18653/v1/n18-2097},
   DOI={10.18653/v1/n18-2097},
   journal={Proceedings of the 2018 Conference of the North American Chapter of
          the Association for Computational Linguistics: Human Language
          Technologies, Volume 2 (Short Papers)},
   publisher={Association for Computational Linguistics},
   author={Cohan, Arman and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Kim, Seokhwan and Chang, Walter and Goharian, Nazli},
   year={2018}
}
"""

_DESCRIPTION = """
A summarization dataset comprised of pairs of scientific papers.
The dataset provides a challenging testbed for abstractive summarization.
It contains papers and their abstracts.
"""

_HOMEPAGE = "https://github.com/armancohan/long-summarization"

_LICENSE = "Apache-2.0 License"

_URL = "https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip"


class SummertimeArxiv(datasets.GeneratorBasedBuilder):
    """Arxiv long summarization dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "article_id": datasets.Value("string"),
                "article_text": [datasets.Value("string")],
                "abstract_text": [datasets.Value("string")],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URL
        path = dl_manager.download_and_extract(my_urls)
        path = os.path.join(path, "arxiv-dataset")

        trainpath = os.path.join(path, "train.txt")
        valpath = os.path.join(path, "val.txt")
        testpath = os.path.join(path, "test.txt")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": trainpath, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": valpath, "split": "val"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": testpath, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        with open(filepath, "r") as f:
            for line in f:

                instance = json.loads(line)

                entry = {}
                entry["article_id"] = instance["article_id"]
                entry["article_text"] = instance["article_text"]
                entry["abstract_text"] = instance["abstract_text"]

                yield entry["article_id"], entry
