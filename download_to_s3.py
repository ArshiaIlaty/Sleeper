#!/usr/bin/env python3
"""Stream PhysioNet Challenge 2026 dataset from Kaggle directly to S3."""

import argparse
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from kagglesdk.datasets.types.dataset_api_service import (
    ApiDownloadDatasetRequest,
    ApiGetDatasetRequest,
)

from kagglehub.clients import build_kaggle_client
from kagglehub.exceptions import handle_call
from kagglehub.handle import parse_dataset_handle


def stream_kaggle_dataset_to_s3(handle: str, bucket: str, key: str) -> None:
    h = parse_dataset_handle(handle)

    with build_kaggle_client() as api_client:
        if not h.is_versioned():
            get_req = ApiGetDatasetRequest()
            get_req.owner_slug = h.owner
            get_req.dataset_slug = h.dataset
            dataset = handle_call(
                lambda: api_client.datasets.dataset_api_client.get_dataset(get_req),
                h,
            )
            h = h.with_version(dataset.current_version_number)

        dl_req = ApiDownloadDatasetRequest()
        dl_req.owner_slug = h.owner
        dl_req.dataset_slug = h.dataset
        dl_req.dataset_version_number = h.version

        response = handle_call(
            lambda: api_client.datasets.dataset_api_client.download_dataset(dl_req),
            h,
        )

        content_length = response.headers.get("Content-Length")
        print(f"Streaming dataset version {h.version} to s3://{bucket}/{key}")
        if content_length:
            print(f"Size: {int(content_length) / 1e9:.2f} GB")

        response.raw.decode_content = False
        s3 = boto3.client("s3")
        config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
        )
        try:
            s3.upload_fileobj(response.raw, bucket, key, Config=config)
        finally:
            response.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handle",
        default="physionet/physionetchallenge2026data",
        help="Kaggle dataset handle",
    )
    parser.add_argument("--bucket", default="physionet2026", help="S3 bucket name")
    parser.add_argument(
        "--key",
        default="physionetchallenge2026data/physionetchallenge2026data.zip",
        help="S3 object key",
    )
    args = parser.parse_args()

    stream_kaggle_dataset_to_s3(args.handle, args.bucket, args.key)
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
