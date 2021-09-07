import urllib.request
import progressbar
import os

from pathlib import Path

# use huggingface cache location here
HF_CACHE_LOCATION = os.environ.get(
    "HF_HOME", Path(Path.home(), ".cache/huggingface")
)

def get_cached_file_path(sub_dir: str, file_name: str, url: str = None, force_download: bool = False):
    """ get the file from cached location, optionally, if not cached, then download it """
    cache_dir = Path(HF_CACHE_LOCATION, sub_dir)

    if not cache_dir.exists():
        cache_dir.mkdir()

    file_path = Path(cache_dir, file_name)
    if file_path.exists() and not force_download:
        return file_path
    elif not file_path.exists() and not url:
        raise ValueError("file {} does not exist and url for downloading is not provided".format(file_path))
    else:
        download_with_progressbar(url, file_path)
        return file_path

pbar = None
def download_with_progressbar(url: str, file_path: str):

    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None

    print(f"Start downloading {file_path} from {url}...")
    urllib.request.urlretrieve(url, file_path, show_progress)