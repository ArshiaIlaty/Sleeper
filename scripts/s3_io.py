"""Shared S3 helpers for PhysioNet 2026 Challenge data (no local full-dataset storage)."""

from __future__ import annotations

import io
import json
import os
from typing import Any, BinaryIO, Dict, Iterable, Optional

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("PHYSIONET_S3_BUCKET", "physionet2026")
ZIP_KEY = os.environ.get(
    "PHYSIONET_ZIP_KEY",
    "physionetchallenge2026data/physionetchallenge2026data.zip",
)
EXTRACT_PREFIX = os.environ.get(
    "PHYSIONET_EXTRACT_PREFIX",
    "physionetchallenge2026data/extracted/training_set_small/",
)
FEATURE_PREFIX = os.environ.get(
    "PHYSIONET_FEATURE_PREFIX",
    "physionetchallenge2026data/features/",
)
MANIFEST_KEY = EXTRACT_PREFIX.rstrip("/") + "/_manifest.json"


def s3_client():
    return boto3.client("s3")


def normalize_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


class S3SeekableReader(io.IOBase):
    """Seekable reader over an S3 object via HTTP Range requests (for zipfile)."""

    def __init__(self, bucket: str, key: str, chunk_size: int = 8 * 1024 * 1024):
        self._s3 = s3_client()
        self._bucket = bucket
        self._key = key
        self._chunk_size = chunk_size
        head = self._s3.head_object(Bucket=bucket, Key=key)
        self._size = head["ContentLength"]
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        if not (self._buf_start <= self._pos < self._buf_start + len(self._buf)):
            self._buf = b""
            self._buf_start = self._pos
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        end = self._size if n < 0 else min(self._pos + n, self._size)
        out = bytearray()
        while self._pos < end:
            if not (self._buf_start <= self._pos < self._buf_start + len(self._buf)):
                fetch_start = self._pos
                fetch_end = min(self._size, fetch_start + self._chunk_size) - 1
                resp = self._s3.get_object(
                    Bucket=self._bucket,
                    Key=self._key,
                    Range=f"bytes={fetch_start}-{fetch_end}",
                )
                self._buf = resp["Body"].read()
                self._buf_start = fetch_start
            take = min(end - self._pos, len(self._buf) - (self._pos - self._buf_start))
            out.extend(self._buf[self._pos - self._buf_start : self._pos - self._buf_start + take])
            self._pos += take
        return bytes(out)

    def readable(self) -> bool:
        return True


def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def s3_object_size(bucket: str, key: str) -> Optional[int]:
    try:
        return int(s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception:
        return None


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3_client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def upload_fileobj(bucket: str, key: str, stream: BinaryIO, content_type: str = "application/octet-stream") -> None:
    s3_client().upload_fileobj(stream, bucket, key, ExtraArgs={"ContentType": content_type})


def download_bytes(bucket: str, key: str) -> bytes:
    return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def download_to_temp(bucket: str, key: str, cache_dir: str = "/home/ec2-user/Sleeper/.s3_cache") -> str:
    """Download one S3 object to a temp cache path (reused if present and size matches)."""
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = key.replace("/", "__")
    local = os.path.join(cache_dir, safe_name)
    remote_size = s3_object_size(bucket, key)
    if remote_size is not None and os.path.exists(local) and os.path.getsize(local) == remote_size:
        return local
    tmp = local + ".part"
    s3_client().download_file(bucket, key, tmp)
    os.replace(tmp, local)
    return local


def load_json(bucket: str, key: str) -> Dict[str, Any]:
    return json.loads(download_bytes(bucket, key).decode("utf-8"))


def save_json(bucket: str, key: str, payload: Dict[str, Any]) -> None:
    upload_bytes(bucket, key, json.dumps(payload, indent=2).encode("utf-8"), "application/json")


def list_keys(bucket: str, prefix: str) -> Iterable[str]:
    prefix = normalize_prefix(prefix)
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                yield key
