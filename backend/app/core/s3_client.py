"""
S3 client utilities for loading data files from AWS S3.

Provides functions to download and load various file types:
- Parquet files (directly into pandas DataFrame)
- Pickle files (into Python objects)
- FAISS index files (download to temp, then load)
"""

import io
import pickle
import tempfile
from typing import Any

import boto3
import faiss
import pandas as pd
from botocore.exceptions import ClientError

from backend.app.core.config import settings


def get_s3_client():
    """
    Create and return a boto3 S3 client.

    Uses credentials from environment variables if provided,
    otherwise falls back to IAM role or AWS CLI profile.
    """
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        return boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    else:
        # Use default credential chain (IAM role, AWS CLI profile, etc.)
        return boto3.client("s3", region_name=settings.aws_region)


def load_parquet_from_s3(s3_key: str) -> pd.DataFrame:
    """
    Load a parquet file from S3 directly into a pandas DataFrame.

    Args:
        s3_key: The S3 object key (path within the bucket).

    Returns:
        pandas DataFrame with the parquet data.
    """
    s3 = get_s3_client()
    bucket = settings.s3_bucket

    print(f"[S3] Loading parquet from s3://{bucket}/{s3_key}")

    response = s3.get_object(Bucket=bucket, Key=s3_key)
    parquet_bytes = response["Body"].read()

    return pd.read_parquet(io.BytesIO(parquet_bytes))


def load_pickle_from_s3(s3_key: str) -> Any:
    """
    Load a pickle file from S3 into a Python object.

    Args:
        s3_key: The S3 object key (path within the bucket).

    Returns:
        The unpickled Python object.
    """
    s3 = get_s3_client()
    bucket = settings.s3_bucket

    print(f"[S3] Loading pickle from s3://{bucket}/{s3_key}")

    response = s3.get_object(Bucket=bucket, Key=s3_key)
    pickle_bytes = response["Body"].read()

    return pickle.loads(pickle_bytes)


def load_faiss_from_s3(s3_key: str) -> faiss.Index:
    """
    Load a FAISS index from S3.

    FAISS requires the index file on disk, so we download to a temp file first.

    Args:
        s3_key: The S3 object key (path within the bucket).

    Returns:
        The loaded FAISS index.
    """
    s3 = get_s3_client()
    bucket = settings.s3_bucket

    print(f"[S3] Loading FAISS index from s3://{bucket}/{s3_key}")

    # Download to a temporary file (FAISS needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp_file:
        s3.download_file(bucket, s3_key, tmp_file.name)
        index = faiss.read_index(tmp_file.name)

    return index


def check_s3_connection() -> bool:
    """
    Test S3 connection by listing the bucket.

    Returns:
        True if connection successful, False otherwise.
    """
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=settings.s3_bucket)
        print(f"[S3] Successfully connected to bucket: {settings.s3_bucket}")
        return True
    except ClientError as e:
        print(f"[S3] Connection failed: {e}")
        return False
