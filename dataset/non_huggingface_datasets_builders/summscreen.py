import os
import json
import datasets


"""Summscreen dataset."""


_CITATION = """
@article{DBLP:journals/corr/abs-2104-07091,
  author    = {Mingda Chen and
               Zewei Chu and
               Sam Wiseman and
               Kevin Gimpel},
  title     = {SummScreen: {A} Dataset for Abstractive Screenplay Summarization},
  journal   = {CoRR},
  volume    = {abs/2104.07091},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.07091},
  archivePrefix = {arXiv},
  eprint    = {2104.07091},
  timestamp = {Mon, 19 Apr 2021 16:45:47 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-07091.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
A summary of scientific papers should ideally incorporate the impact of the papers on the research community 
reflected by citations. To facilitate research in citation-aware scientific paper summarization (Scisumm), 
the CL-Scisumm shared task has been organized since 2014 for papers in the computational linguistics and NLP domain.
"""

_HOMEPAGE = "https://github.com/mingdachen/SummScreen"

_LICENSE = "MIT Licencse"

_URLs = 'https://drive.google.com/uc?id=1BvdIllGBo9d2-bzXQRzWuJXB04XPVmfF'




class SummertimeSummscreen(datasets.GeneratorBasedBuilder):
    """Summscreen dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(),
    ]

    def _info(self):
        features = datasets.Features(
            {
                'entry_number': datasets.Value("string"),
                'transcript': datasets.features.Sequence(datasets.Value("string")),
                'recap': datasets.Value("string")
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
        path = os.path.join(path, "SummScreen")

        trainpath_fd = os.path.join('ForeverDreaming', 'fd_train.json')
        trainpath_tms = os.path.join('TVMegaSite', 'tms_train.json')
        trainpaths = [trainpath_fd, trainpath_tms]

        devpath_fd = os.path.join('ForeverDreaming', 'fd_dev.json')
        devpath_tms = os.path.join('TVMegaSite', 'tms_dev.json')
        devpaths = [devpath_fd, devpath_tms]

        testpath_fd = os.path.join('ForeverDreaming', 'fd_test.json')
        testpath_tms = os.path.join('TVMegaSite', 'tms_test.json')
        testpaths = [testpath_fd, testpath_tms]


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepaths": (path, trainpaths), "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepaths": (path, devpaths), "split": "dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepaths": (path, testpaths), "split": "test"}
            )
        ]

    def _generate_examples(self, filepaths, split):
        """Yields examples."""

        counter = 0
        path, relative_filepaths = filepaths
        for filepath in relative_filepaths:

            extraction_path = os.path.join(path, filepath)

            with open(extraction_path, 'r') as f:
                for line in f:
                    processed_line = line.replace("@@ ", "")
                    instance = json.loads(processed_line)

                    entry = {}
                    entry['entry_number'] = instance['filename']
                    entry['transcript'] = instance['Transcript']
                    entry['recap'] = instance['Recap'][0]    # Recap is a single string in list

                    yield entry['entry_number'], entry
        