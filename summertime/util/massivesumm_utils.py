"""
Copyright 2018 Max Grusky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This this file has been altered and adopted. Specifically, it is modified to accomodate
the common crawl archive instead of the waybackmachine.
"""

import re

from urllib.parse import quote, urlparse, urljoin
from bs4 import BeautifulSoup
from readability import Document
import requests

import argparse
import gzip
import io
import os
import json
import orjson
import random
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

_whitespace = re.compile(r"\s+")

class Article(object):

    """
    Reads in a {url: "", html: ""} archive entry from the downloader script.
    This will scrape the provided HTML and extract the summary and text. Note
    that the provided URL in this case is actually the ARCHIVE url (Maybe this
    should be made clearer in the downloader script?).
    """

    def __init__(self, archive, html):

        self.archive = archive
        self.html = html if html is not None else ""

        # @djam my doing
        self.url = archive
        self.date = None
        # self._parse_archive()
        self._parse_html()

    def _parse_archive(self):

        *splits, url = self.archive.split("id_/")
        *_, date = splits[0].split("/")

        self.url = self.normalize_url(url)
        self.date = date

    def _parse_html(self):

        self._load_html()
        self._find_canonical_url()

        self._extract_text()
        self._extract_summary()

    def _extract_summary(self):

        self.all_summaries = {}

        for meta in self.soup.findAll("meta"):
            for attr, value in meta.attrs.items():

                if attr in ("name", "property") and "description" in value:

                    # Extract the tag content. If we can't find anything,
                    # ignore it and move onto the next tag.

                    try:

                        self.all_summaries[value] = meta.get("content").strip()

                    except Exception:

                        continue

        if len(self.all_summaries) == 0:

            self.summary = None
            return

        for kind in ("og:description", "twitter:description", "description"):

            if kind in self.all_summaries:

                self.summary = self.all_summaries[kind]
                break

        else:

            random_pick = sorted(self.all_summaries)[0]
            self.summary = self.all_summaries[random_pick]

    def _extract_text(self):

        """
        Uses Readability to extract the body text and titles of the articles.
        """

        # Confusingly, the Readability package calls the body text of the article
        # its "summary." We want to create a plain text document from the body text,
        # so we need to extract the text from Readability's HTML version.

        body_soup = BeautifulSoup(self.readability.summary(), "lxml")

        # Now go through and extract each paragraph (in order).

        paragraph_text = []
        for paragraph in body_soup.findAll("p"):

            # Very short pieces of text tend not to be article body text, but
            # captions, attributions, and advertising. It seems that excluding
            # paragraphs shorter than five words removes most of this.

            if len(paragraph.text.split()) >= 5:

                paragraph_body = _whitespace.sub(" ", paragraph.text).strip()
                paragraph_text.append(paragraph_body)

        # We join the plain text paragraphs of the article with double new lines.

        self.text = "\n\n".join(paragraph_text)

        # "Short title" uses in-page heuristics to remove cruft from <title>; e.g.:
        # .title():       American Recalls Moment Leg Broken by Truck in Nice - ABC News
        # .short_title(): American Recalls Moment Leg Broken by Truck in Nice

        self.title = self.readability.short_title()

    def _load_html(self):

        # Readability crashes if it encounters empty pages.

        if self.html.strip() == "":

            raise Exception("No page content?")

        # The document has content. Create:
        # - A Readability parse object to extract the text
        # - A full-page BeautifulSoup object to extract summaries.

        self.readability = Document(self.html)
        self.soup = BeautifulSoup(self.html, "lxml")

    def _find_canonical_url(self):

        # Start out by normalizing the URL as we know it. Without reading the
        # page yet, this is our best guess of the article's canonical URL.

        self.original_url = self.url

        try:

            # Try to extract the page's canonical URL, if it has one. If it doesn't,
            # BeautifulSoup will raise an exception, and we will give up, sticking
            # with the normalized URL as the best URL.

            rel_canon = self.soup.find("link", {"rel": "canonical"}).get("href")

            # I've sometimes seen the canonical URL be relative to the current page.
            # Although this is rare, we can handle this using our best knowledge of
            # the page's URL so far. Just in case, we'll normalize this too.

            abs_canon_url = urljoin(self.url, rel_canon)
            norm_canon_url = self.normalize_url(abs_canon_url)

            # Sometimes, the canonical URL will be on a completely different domain.
            # I'm not sure why. But as a sanity check, make sure it's on the same
            # domain before using it.

            if self.same_domain(self.url, norm_canon_url):

                self.url = self.norm_canon_url

        except Exception:

            # If we've failed at some point (most likely because the page doesn't
            # use the canonical tag), set the canonical and normalized canonical
            # URLs to None so that the user is aware of this.

            pass

    def serialize(self):

        """
        Return simple page object to JSONify and write to file.
        """

        return {
            "url": self.url,
            "archive": self.archive,
            "title": self.title,
            "date": self.date,
            "text": self.text,
            "summary": self.summary,
        }

    @staticmethod
    def process(page):

        url = page.get("archive", page.get("url"))
        html = page.get("html", "")
        if html is None:
            html = ""

        try:
            return Article(url, html).serialize()
        except:
            print("FAILING TO PROCESS HTML")
            return None

    @staticmethod
    def same_domain(url1, url2):

        """
        Check if two URLs share the same domain (urlparse netloc).
        This is used primarily in evaluating canonical URLs.
        """

        return urlparse(url1).netloc == urlparse(url2).netloc

    @staticmethod
    def normalize_url(url):

        """
        Remove fragments, ports, and other junk from Archive.org scrapes.
        This is to detect duplicate pages, and prettify URLs.
        """

        # Multiple forward slashes should be replaced with just one.

        cleaned = url.replace("://", "\0").replace("//", "/").replace("\0", "://")

        # Removing fragments and query parameters.

        parsed = urlparse(cleaned)
        parsed = parsed._replace(
            path=quote(parsed.path, safe="%/"),
            netloc=parsed.netloc.replace(":80", ""),
            query="",
            fragment="",
        )

        return parsed.geturl()

def load_samples(filename: str) -> list:
    samples = []
    with gzip.open(filename) as fh_in:
        for row in tqdm(fh_in):
            sample = json.loads(row)
            samples.append(sample)
    return samples


def download_sample(sample: dict) -> dict:
    filename = sample["filename"]
    length = int(sample["length"])
    offset = int(sample["offset"])

    offset_end = offset + length - 1
    # We'll get the file via HTTPS so we don't need to worry about S3 credentials
    # Getting the file on S3 is equivalent however - you can request a Range
    prefix = "https://commoncrawl.s3.amazonaws.com/"
    # We can then use the Range header to ask for just this set of bytes
    try:
        resp = requests.get(
            prefix + filename,
            headers={"Range": "bytes={}-{}".format(offset, offset_end)},
        )

        compressed_file = io.BytesIO(resp.content)
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        data = decompressed_file.read().decode()
        warc, header, response = data.strip().split("\r\n\r\n", 2)
        return {"html": response, **sample}
    except:
        with open("error.log", "at") as err_log:
            err_log.write(json.dumps(sample) + "\n")
        return None


def download_list(samples, n_processes: int):
    with Pool(n_processes) as pool:
        for sample in pool.imap_unordered(download_sample, samples):
            if sample:  # don't yield failing samples
                yield sample


def run(url_file: str):
    n_proc = cpu_count()
    limit = -1

    samples = load_samples(url_file)
    if limit > 0:
        samples = random.sample(samples, limit)

    downloaded = []
    for sample in tqdm(
        download_list(samples, n_processes=n_proc), total=len(samples)
    ):
        downloaded.append(json.dumps(sample))

    return downloaded


def extract(archive):
    n_proc = cpu_count()
    batch_size = n_proc * 20
    # previously = set()
    todo = set()

    # if os.path.isfile(dataset):

    #     print("Comparing archive and dataset files: ", end="")

    #     with gzip.open(dataset) as dataset_file:

    #         for article in dataset_file:
    #             article = orjson.loads(article)
    #             url = article.get("archive", article.get("url"))
    #             previously.add(url)

    #     print("found", len(previously), "finished summaries... ", end="")
    # else:
    print("Loading downloaded summaries: ", end="")

    # with gzip.open(archive) as archive_file:
    for article in archive:
        article = orjson.loads(article)
        url = article.get("archive", article.get("url"))
        todo.add(url)

    # todo -= previously

    print("found", len(todo), "new summaries to extract.\n")
    dataset = []

    with tqdm(total=len(todo), desc="Extracting Summaries") as progress:
        # with gzip.open(archive) as archive_file:
            # with gzip.open(dataset, "at") as dataset_file:

        chunk = []

        def process_batch():

            with Pool(n_proc) as ex:
                results = list(ex.map(Article.process, chunk))
                results = [r for r in results if r is not None]

                for result in results:
                    if result["text"] is None or result["summary"] is None:
                        continue
                    else:
                        dataset.append(json.dumps(result))

                progress.update(len(results))

        for article in archive:
            article = orjson.loads(article)
            url = article.get("archive", article.get("url"))
            if url not in todo:
                continue

            chunk.append(article)
            
            if len(chunk) >= batch_size:
                process_batch()
                chunk = []

        process_batch()

    print("\nExtraction complete.")
    return dataset

def massivesumm_extract_from_url(urls):

    archive = run(urls) #archive should be eliminated as a parameter (make this function return a list of dict objects instead of writing to a gzipped jsonl

    dataset = extract(archive) #dataset arg should also be eliminated as a parameter (return a list of dict objects instead of writing to a gzipped jsonl)

    return dataset