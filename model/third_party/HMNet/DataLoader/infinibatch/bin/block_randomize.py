# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/python3.6

# simple command-line wrapper around the chunked_dataset_iterator
# Example:
#   block_randomize my_chunked_data_folder/
#   block_randomize --azure-storage-key $MY_KEY https://myaccount.blob.core.windows.net/mycontainer/my_chunked_data_folder

import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))  # find our imports

from infinibatch.datasets import chunked_dataset_iterator

from typing import Union, Iterator, Callable, Any, Optional, Dict
import os, sys, re
import gzip


# helper functions to abstract access to Azure blobs
# @TODO: These will be abstracted into a helper library in a future version.
def _try_parse_azure_blob_uri(path: str):
    try:
        m = re.compile("https://([a-z0-9]*).blob.core.windows.net/([^/]*)/(.*)").match(path)
        #print (m.group(1))
        #print (m.group(2))
        #print (m.group(3))
        return (m.group(1), m.group(2), m.group(3))
    except:
        return None


def _get_azure_key(storage_account: str, credentials: Optional[Union[str,Dict[str,str]]]):
    if not credentials:
        return None
    elif isinstance(credentials, str):
        return credentials
    else:
        return credentials[storage_account]


def read_utf8_file(path: str, credentials: Optional[Union[str,Dict[str,str]]]) -> Iterator[str]:
    blob_data = _try_parse_azure_blob_uri(path)
    if blob_data is None:
        with open(path, "rb") as f:
            data = f.read()
    else:
        try:
            # pip install azure-storage-blob
            from azure.storage.blob import BlobClient
        except:
            print("Failed to import azure.storage.blob. Please pip install azure-storage-blob", file=sys.stderr)
            raise
        data = BlobClient.from_blob_url(path, credential=_get_azure_key(storage_account=blob_data[0], credentials=credentials)).download_blob().readall()
    if path.endswith('.gz'):
        data = gzip.decompress(data)
    # @TODO: auto-detect UCS-2 by BOM
    return iter(data.decode(encoding='utf-8').splitlines())


def enumerate_files(dir: str, ext: str, credentials: Optional[Union[str,Dict[str,str]]]):
    blob_data = _try_parse_azure_blob_uri(dir)
    if blob_data is None:
        return [os.path.join(dir, path.name)
                for path in os.scandir(dir)
                if path.is_file() and (ext is None or path.name.endswith(ext))]
    else:
        try:
            # pip install azure-storage-blob
            from azure.storage.blob import ContainerClient
        except:
            print("Failed to import azure.storage.blob. Please pip install azure-storage-blob", file=sys.stderr)
            raise
        account, container, blob_path = blob_data

        print("enumerate_files: enumerating blobs in", dir, file=sys.stderr, flush=True)
        # @BUGBUG: The prefix does not seem to have to start; seems it can also be a substring
        container_uri = "https://" + account + ".blob.core.windows.net/" + container
        container_client = ContainerClient.from_container_url(container_uri, credential=_get_azure_key(account, credentials))
        if not blob_path.endswith("/"):
            blob_path += "/"
        blob_uris = [container_uri + "/" + blob["name"] for blob in container_client.walk_blobs(blob_path, delimiter="") if (ext is None or blob["name"].endswith(ext))]
        print("enumerate_files:", len(blob_uris), "blobs found", file=sys.stderr, flush=True)
        for blob_name in blob_uris[:10]:
            print(blob_name, file=sys.stderr, flush=True)
        return blob_uris


if sys.argv[1] == "--azure-storage-key":
    credential = sys.argv[2]
    paths = sys.argv[3:]
else:
    credential = None
    paths = sys.argv[1:]

chunk_file_paths = [  # enumerate all .gz files in the given paths
    subpath
    for path in paths
    for subpath in enumerate_files(path, '.gz', credential)
]
chunk_file_paths.sort()  # make sure file order is always the same, independent of OS
print("block_randomize: reading from", len(chunk_file_paths), "chunk files", file=sys.stderr)

ds = chunked_dataset_iterator(chunk_refs=chunk_file_paths, read_chunk_fn=lambda path: read_utf8_file(path, credential),
                              shuffle=True, buffer_size=1000000, seed=1, use_windowed=True)
for line in ds:
    print(line)
