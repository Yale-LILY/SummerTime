import os
import json
import gdown
import datasets
from summertime.util.massivesumm_utils import massivesumm_extract_from_url

"""Massivesumm Dataset."""


_CITATION = """
@inproceedings{varab-schluter-2021-massivesumm,
    title = "{M}assive{S}umm: a very large-scale, very multilingual, news summarisation dataset",
    author = "Varab, Daniel  and
      Schluter, Natalie",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.797",
    pages = "10150--10161",
    abstract = "Current research in automatic summarisation is unapologetically anglo-centered{--}a persistent state-of-affairs, which also predates neural net approaches. High-quality automatic summarisation datasets are notoriously expensive to create, posing a challenge for any language. However, with digitalisation, archiving, and social media advertising of newswire articles, recent work has shown how, with careful methodology application, large-scale datasets can now be simply gathered instead of written. In this paper, we present a large-scale multilingual summarisation dataset containing articles in 92 languages, spread across 28.8 million articles, in more than 35 writing scripts. This is both the largest, most inclusive, existing automatic summarisation dataset, as well as one of the largest, most inclusive, ever published datasets for any NLP task. We present the first investigation on the efficacy of resource building from news platforms in the low-resource language setting. Finally, we provide some first insight on how low-resource language settings impact state-of-the-art automatic summarisation system performance.",
}
"""

_DESCRIPTION = """
A massively multilingual summarization dataset.
Consists of 12.3 million articles and their summaries
across 92 languages and 370 news platforms.

"""

_HOMEPAGE = "https://github.com/danielvarab/massive-summ"

_LICENSE = "Apache-2.0 License"

# dict of language: drive id
_URL_IDs = {
    "afrikaans": "",
    "amharic": "1_awz_-B0iWtaPdKih8H8Kz4HGnJSwvRq",
    "arabic": "1HvCeJ3p59sdhb1xLFGNVHr10hHEg8phA",
    "assamese": "",
    "aymara": "",
    "azerbaijani": "1CftuzziqiR5QezH9oYL-bCE_KpKdKQgK",
    "bambara": "1MWQVJMBLmc_8qktep7FGohVbHdXR0iKx",
    "bengali": "1wK6YTRkXuc4df8C-Ko-PaWB1pIeDfY8q",
    "tibetan": "1vnZb9PUjRCX6E__OlCAqBGxdu3-19Q8W",
    "bosnian": "1TTQVPZ4G7TGy7mFnN21XC3ZDlJlTpGhM",
    "bulgarian": "13MJzUdrCLz-lo_c4IZOOupJY_50zvZHd",
    "catalan": "",
    "czech": "1tKzsoGFdDo93aKfEpkY5sSLsuN1hL4LV",
    "welsh": "1ewLaDdoC1An4hYr6LVLnPGPsY5h0ZiqS",
    "danish": "1-VcQxG_YDngEaNNRBMn9vl6L3sIEnL_8",
    "german": "1LfBPlYTbmjWnZTM_e6twUVgzrOjY2kfp",
    "greek": "1dzbQc2K_rTIrkpcw9UgQyZ4PPk_ZYF5i",
    "english": "1u-Zt56FKrJ9zVZRRSPiwqIoGaKHkTOuY",
    "esperanto": "",
    "persian": "1AMz5xhJaR9Ud-oic4LA4-VoBfcT-cqWH",
    "filipino": "",
    "french": "1UQitwbOPwbbaXFb8xtV0chjeLvWzm3LN",
    "fulah": "1eku0kULX4ZE9wQnUJMcqHy65Jsu21FFp",
    "irish": "1HNVxzYdmc1l_q4UOwNV6Yh2_QMfmlSaf",
    "gujarati": "1PWFIVGeCRuzAHH-w2UwVOANSvyeEXgFk",
    "haitian": "1yDorOERjCNFdDRt9viZyWpcO7gmXDyvr",
    "hausa": "1cYkwEYclvHnN8BLZf6z-DEyINGAHy34L",
    "hebrew": "",
    "hindi": "1ZNcCqUV15Bv2FlY3qkMYyBWBm0hO4LKI",
    "croatian": "",
    "hungarian": "1R52kqwahdPHFkGpsGdAJE6Wkq38UGgFS",
    "armenian": "1ciACol27dN07_omNInoU_NqUvYmwXo6C",
    "igbo": "1oYOHwATB0PWNYvEv-azy_8MUgkxzFZCY",
    "indonesian": "1Sch920J5PqJbhpEQHNTjJ1ojiMU46tiQ",
    "icelandic": "1wzccfq0RAN7c5c2BGhNMySp2yMYfj9ep",
    "italian": "123eRVzORxPIQnp75RMf0LsXWL21l76IH",
    "japanese": "1vjYBbEmWg8PoztrcSqDjUe7ClCUNKHAL",
    "kannada": "18rBERL7l4zBupWwVHXasPu3jlegCM31B",
    "georgian": "1GSpqPf87onRlKHu4yoLzxQkAOSIE1GVW",
    "khmer": "1-0m54dcSjGyBST9bodw1RJYqICsZCwuS",
    "kinyarwanda": "",
    "kyrgyz": "1ncixaRUVSGcgTrMPhibN1Pfd4yIJ8c15",
    "korean": "14-QZft00ab2KAtjT1-p1fvaA45qKt3JJ",
    "kurdish": "1g3WTVRxMo5M5HOBuNJLU7KdSw1RQRdTx",
    "lao": "1IOcXBGMoaA859RXzrXSV1WS2qMOweRUn",
    "latvian": "1AdXmbWraGH_Dh9_f2CcQhhqP5hIqnJXu",
    "lingala": "1QDYxfhMQDZeGVVRUsZjUzf7C2RUMYDMR",
    "lithuanian": "1WjIJ-LZN0ZdqtE_NnoEiAiNhXm3eRGHk",
    "malayalam": "1tvsdnjRAiBFHc0Py-duJoqPlSwYlokie",
    "marathi": "1Z-ui3IipNQy3jpQqeNzrQiVZcXnUFs2e",
    "macedonian": "1xpE3nPcs-m5WdbPyOX1H4wt9k06NMKN3",
    "malagasy": "11mpExgMv7VSdejMXUQLFnnPwitLUNudw",
    "mongolian": "1rejguZ0HNNMZdV_9g6qXT6Si6QVXhuge",
    "burmese": "1cLre9C9f1lm2Ds_8hv6f7h2R4phrXtVd",
    "south ndebele": "1KxB5RLGMlteQOqBYu2DOhIXVzCUkhvDM",
    "nepali": "1jw_P1wenbskDfG8iQD3dYRb0oWnRba9N",
    "dutch": "",
    "oriya": "1a3t0X7PfphDZJiyJSL-FzsoZsDH9KIu4",
    "oromo": "1-SopeFs8niXlmwWSe117-YDQ6ECK8xTh",
    "punjabi": "1t3sUOR_m4blOj8iIU1q8ohxUWPTWImcw",
    "polish": "1pNOSculzyCNMjrQOarVhG_SDg1IbMpBr",
    "portuguese": "13ET2tIsrzFTzlb9Rd2KAp-FF7Y6R2Ker",
    "dari": "11QMxXjH9vN0V6-UZXT2omxb7lm8zphqC",
    "pashto": "1nccq6pEsvUhe1zvPTDoDKEob6cRKPpWP",
    "romanian": "",
    "rundi": "165N8Wh_TeTo7N_el6eWGmZ5ts3KBKU9Q",
    "russian": "15Cqcrbl_lG_oSED_hTyR_mb-dwpvw9J8",
    "sinhala": "1gvqSIOkL7RDX-yg7O1VwHTdRUcNGcf_F",
    "slovak": "1GDzuxd-KhBA_fHrDlO8HcCayLsG4-UkX",
    "slovenian": "1T71uVRLX-wB-qeWFMyqtxlI91Dr2u7pn",
    "shona": "1wCysDOCUvsA3H-9CmrItwU7GHIxQt1rU",
    "somali": "1oXDsB76ViX9ri5W_2xVEQqLWDqIQAh5O",
    "spanish": "14dX8cePpcb-E7nS8brupu9lQkoLUPLSf",
    "albanian": "1jz5sXn8JeHhZHjXb7r0wLqyynva7c8Al",
    "serbian": "1XJCyan1OL3UTI9_tNbvZX2ngQ2bhoy0U",
    "swahili": "1ukbOFz_dHYaIQnD4ub0hE9DqF_1mwVke",
    "swedish": "",
    "tamil": "1DWv-hkU0P2B0AysTFwOZ3aghW6lxF16A",
    "telugu": "1e0KIPqcKYHXmSjgLpxLJkEFumw8Bae_g",
    "tetum": "",
    "tajik": "",
    "thai": "197vyuI2JzOGczeVRqnUGu78G3T2WWRit",
    "tigrinya": "1lDazmqixV4Gem96O-c-gulqSNKUEcjnR",
    "turkish": "1Kole41CnnNArIt_rxfNlimk9EMQZFa8Y",
    "ukrainian": "1H8TUR73sJs_bvjJuLNB4szqiCIvTq5sp",
    "urdu": "1HDwEMuaULkZr6Mm39CifS2szyI_vql-G",
    "uzbek": "1nYOLG5UlV-YDeex8Tvi37hK4pM9wD7Wg",
    "vietnamese": "1uts1nSGwWNxEFZnsJi6SdamG2DAVPq-q",
    "xhosa": "1P31PeL7cVJ9eNE-YZ0ofH0pozT9ta5bP",
    "yoruba": "17ifvygtGzaIgDuqiFK0QDK1Jd7SOnkNd",
    "yue chinese": "1blW_lXnUFa3poUwR6YuHd3N2fVoGuHhG",
    "chinese": "13-qysDM2uAiT_E9KjAKsQZAQP_-0pyRL",
    "bislama": "1zUn6LDov0zi_hxYbs9hX63UwLhcNcrsa",
    "gaelic": "1rIYRlZQ0Sl6By45hdOozAS_37Dv6LFQp",
}


class SummertimeMassivesumm(datasets.GeneratorBasedBuilder):
    """Massivesumm Dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"{language}",
            version=datasets.Version("1.1.0"),
            description=f"{language} portion of Massivesumm dataset",
        )
        for language in _URL_IDs.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "article_url": datasets.Value("string"),
                "article": datasets.Value("string"),
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
        # get google drive file id
        drive_id = _URL_IDs[self.config.name]
        # get download url for the file
        url = f"https://drive.google.com/uc?id={drive_id}&export=download"

        path = gdown.cached_download(url)

        # download webpages and scrape summaries into json format
        data = massivesumm_extract_from_url(path)

        # get huggingface's cache dir by using download manager to attempt download from url
        data_dir = dl_manager.download(url)
        data_dir = os.path.dirname(data_dir)
        # save the extracted data to the data_dir
        if not os.path.exists(data_dir + "train.jsonl"):
            os.makedirs(data_dir + "train.jsonl")
        with open(data_dir + "train.jsonl", "w+", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)

                entry = {
                    "article_url": data["url"],
                    "article": data["text"],
                    "summary": data["summary"],
                }
                yield entry["article_url"], entry
