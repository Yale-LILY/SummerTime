import os
import json
import datasets


"""QMsum dataset."""


_CITATION = """
@inproceedings{zhong2021qmsum,
   title={{QMS}um: {A} {N}ew {B}enchmark for {Q}uery-based {M}ulti-domain {M}eeting {S}ummarization},
   author={Zhong, Ming and Yin, Da and Yu, Tao and Zaidi, Ahmad and Mutuma, Mutethia and Jha, Rahul and Hassan Awadallah, Ahmed and Celikyilmaz, Asli and Liu, Yang and Qiu, Xipeng and Radev, Dragomir},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
"""

_DESCRIPTION = """
QMSum is a new human-annotated benchmark for query-based multi-domain meeting summarization task, \
which consists of 1,808 query-summary pairs over 232 meetings in multiple domains.
"""

_HOMEPAGE = "https://github.com/Yale-LILY/QMSum"

_BASE_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl"
_URLs = {
    'train': _BASE_URL + "/train.jsonl",
    'val': _BASE_URL + "/val.jsonl",
    'test': _BASE_URL + "/test.jsonl"
}




class SummertimeQmsum(datasets.GeneratorBasedBuilder):
    """QMsum dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(),
    ]

    def _info(self):
        features = datasets.Features(
            {   
                "entry_number": datasets.Value("string"),
                'meeting_transcripts' :
                    [{
                        'speaker' : datasets.Value("string"),
                        'content' : datasets.Value("string")
                    }],
                'general_query_list':
                    [{
                        'query' : datasets.Value("string"),
                        'answer' : datasets.Value("string")
                    }],
                'specific_query_list': 
                    [{
                        'query' : datasets.Value("string"),
                        'answer' : datasets.Value("string"),
                        'relevant_text_span' : [[datasets.Value("string")]]
                    }]  
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license= None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs
        downloaded_files = dl_manager.download_and_extract(my_urls)

        trainpath = downloaded_files['train']
        valpath = downloaded_files['val']
        testpath = downloaded_files['test']


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepath": trainpath, "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepath": valpath, "split": "val"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs = {"filepath": testpath, "split": "test"}
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        extraction_path = os.path.join(filepath)

        with open(extraction_path) as f:
            for i, line in enumerate(f):
    
                instance = json.loads(line)

                entry = {}
                entry['entry_number'] = split + "_" +str(i)
                entry['meeting_transcripts'] = instance['meeting_transcripts']
                entry['general_query_list'] = instance['general_query_list']
                entry['specific_query_list'] = instance['specific_query_list']

                yield entry['entry_number'], entry
