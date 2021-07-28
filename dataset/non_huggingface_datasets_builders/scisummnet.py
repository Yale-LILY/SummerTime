import os
import datasets


"""Scisummnet dataset."""


_CITATION = """
@InProceedings{yasunaga&al.19.scisumm,
    title = {{ScisummNet}: A Large Annotated Corpus and Content-Impact Models for Scientific Paper Summarization with Citation Networks},
    author = {Michihiro Yasunaga and Jungo Kasai and Rui Zhang and Alexander Fabbri and Irene Li and Dan Friedman and Dragomir Radev},
    booktitle = {Proceedings of AAAI 2019},
    year = {2019}
}
@InProceedings{yasunaga&al.17,
  title = {Graph-based Neural Multi-Document Summarization},
  author = {Yasunaga, Michihiro and Zhang, Rui and Meelu, Kshitijh and Pareek, Ayush and Srinivasan, Krishnan and Radev, Dragomir R.},
  booktitle = {Proceedings of CoNLL 2017},
  year = {2017}
}
"""

_DESCRIPTION = """
A summary of scientific papers should ideally incorporate the impact of the papers on the research community 
reflected by citations. To facilitate research in citation-aware scientific paper summarization (Scisumm), 
the CL-Scisumm shared task has been organized since 2014 for papers in the computational linguistics and NLP domain.
"""

_HOMEPAGE = "https://cs.stanford.edu/~myasu/projects/scisumm_net/"

_LICENSE = "CC BY-SA 4.0"

_URLs = "https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip"


class SummertimeScisummnet(datasets.GeneratorBasedBuilder):
    """Scisummnet dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "entry_number": datasets.Value("int64"),
                "document_xml": datasets.Value("string"),
                "citing_sentences_annotated.json": datasets.Value("string"),
                "summary": datasets.Value("string"),
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
        my_urls = _URLs
        path = dl_manager.download_and_extract(my_urls)
        trainpath = os.path.join(path, 'scisummnet_release1.1__20190413', 'top1000_complete')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"extraction_path": trainpath, "split": "train"}
            )
        ]

    def _generate_examples(self,  extraction_path, split):
        """Yields examples."""

        counter = 0
        for folder in os.listdir(extraction_path):
            entry = {}
            entry['entry_number'] = counter
            counter+=1

            doc_xml_path = os.path.join(extraction_path, folder, 'Documents_xml', folder + ".xml")
            with open (doc_xml_path, "r", encoding='utf-8') as f:
                entry['document_xml'] = f.read()

            cite_annot_path = os.path.join(extraction_path, folder, 'citing_sentences_annotated.json')
            with open (cite_annot_path, "r", encoding='utf-8') as f:
                entry['citing_sentences_annotated.json'] = f.read()

            summary_path = os.path.join(extraction_path, folder, 'summary', folder + ".gold.txt")
            with open (summary_path, "r", encoding='utf-8') as f:
                entry['summary'] = f.read()

            yield entry['entry_number'], entry
            