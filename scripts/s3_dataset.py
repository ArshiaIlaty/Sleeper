"""S3-backed dataset access for PhysioNet 2026 (extracted prefix or zip fallback)."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from typing import Dict, Optional, Tuple

import edfio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from helper_code import (  # noqa: E402
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    find_patients,
    load_demographics,
    load_diagnoses,
)

from s3_io import (  # noqa: E402
    BUCKET,
    EXTRACT_PREFIX,
    S3SeekableReader,
    ZIP_KEY,
    download_bytes,
    download_to_temp,
    normalize_prefix,
    s3_key_exists,
)

ZipFileHandle = Optional[zipfile.ZipFile]


def _edf_to_arrays(edf) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    channel_dict: Dict[str, np.ndarray] = {}
    fs_dict: Dict[str, float] = {}
    for sig in edf.signals:
        label = sig.label.lower().strip()
        fs_dict[label] = float(sig.sampling_frequency)
        channel_dict[label] = sig.data
    return channel_dict, fs_dict


def load_edf_bytes(data: bytes):
    try:
        return edfio.read_edf(io.BytesIO(data), lazy_load_data=False)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            return edfio.read_edf(path, lazy_load_data=False)
        finally:
            os.unlink(path)


class S3ChallengeDataset:
    """Read Challenge files from extracted S3 objects, with optional zip fallback."""

    def __init__(
        self,
        bucket: str = BUCKET,
        extract_prefix: str = EXTRACT_PREFIX,
        zip_key: str = ZIP_KEY,
        cache_dir: str = "/home/ec2-user/Sleeper/.s3_cache",
        max_ram_edf_bytes: int = 200 * 1024 * 1024,
    ):
        self.bucket = bucket
        self.extract_prefix = normalize_prefix(extract_prefix)
        self.zip_key = zip_key
        self.cache_dir = cache_dir
        self.max_ram_edf_bytes = max_ram_edf_bytes
        self._zip: ZipFileHandle = None
        self._zip_reader = None

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def _zip_file(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip_reader = S3SeekableReader(self.bucket, self.zip_key)
            self._zip = zipfile.ZipFile(self._zip_reader)
        return self._zip

    def member_key(self, member_name: str) -> str:
        return self.extract_prefix + member_name.lstrip("/")

    def has_extracted(self, member_name: str) -> bool:
        return s3_key_exists(self.bucket, self.member_key(member_name))

    def read_member_bytes(self, member_name: str) -> bytes:
        key = self.member_key(member_name)
        if self.has_extracted(member_name):
            return download_bytes(self.bucket, key)
        return self._zip_file().read(member_name)

    def load_edf_member(self, member_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        if self.has_extracted(member_name):
            data = download_bytes(self.bucket, self.member_key(member_name))
            if len(data) > self.max_ram_edf_bytes:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    path = tmp.name
                    tmp.write(data)
                try:
                    edf = edfio.read_edf(path, lazy_load_data=False)
                finally:
                    if os.path.exists(path):
                        os.unlink(path)
            else:
                edf = load_edf_bytes(data)
        else:
            info = self._zip_file().getinfo(member_name)
            if info.file_size > self.max_ram_edf_bytes:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    path = tmp.name
                try:
                    with self._zip_file().open(member_name, "r") as src, open(path, "wb") as dst:
                        while chunk := src.read(8 * 1024 * 1024):
                            dst.write(chunk)
                    edf = edfio.read_edf(path, lazy_load_data=False)
                finally:
                    if os.path.exists(path):
                        os.unlink(path)
            else:
                edf = load_edf_bytes(self._zip_file().read(member_name))
        return _edf_to_arrays(edf)

    def demographics_bytes(self) -> bytes:
        return self.read_member_bytes(DEMOGRAPHICS_FILE)

    def patient_records(self) -> list:
        demo_path = os.path.join(self.cache_dir, "demographics.csv")
        os.makedirs(self.cache_dir, exist_ok=True)
        data = self.demographics_bytes()
        with open(demo_path, "wb") as f:
            f.write(data)
        return find_patients(demo_path)

    def load_demographics_for_record(self, record: dict) -> dict:
        pid = record[HEADERS["bids_folder"]]
        sess = record[HEADERS["session_id"]]
        demo_path = os.path.join(self.cache_dir, "demographics.csv")
        if not os.path.exists(demo_path):
            with open(demo_path, "wb") as f:
                f.write(self.demographics_bytes())
        try:
            return load_demographics(demo_path, pid, sess)
        except Exception:
            return {}

    def load_label(self, pid: str) -> Optional[int]:
        demo_path = os.path.join(self.cache_dir, "demographics.csv")
        if not os.path.exists(demo_path):
            with open(demo_path, "wb") as f:
                f.write(self.demographics_bytes())
        try:
            return load_diagnoses(demo_path, pid)
        except Exception:
            return None
