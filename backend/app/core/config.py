"""
Centralized configuration for the RAG application.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Storage configuration
    storage_mode: Literal["s3", "local"] = "local"

    # AWS S3 configuration
    s3_bucket: str = "charlie-stamp-rag-pipeline"
    aws_region: str = "eu-west-2"

    # Optional: AWS credentials (if not using IAM roles or AWS CLI profile)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            storage_mode=os.getenv("STORAGE_MODE", "local"),  # type: ignore
            s3_bucket=os.getenv("S3_BUCKET", "charlie-stamp-rag-pipeline"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )


# Global settings instance
settings = Settings.from_env()
