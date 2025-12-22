#!/usr/bin/env python3
"""
Download parquet files from R2 bucket.

Uses existing R2 environment variables from .env file.

Usage:
    # List all parquet files in bucket
    python scripts/download_parquet.py --list

    # Download specific file by hotkey and window
    python scripts/download_parquet.py --hotkey 5H1NNZw... --window 7155690

    # Download to specific directory
    python scripts/download_parquet.py --hotkey 5H1NNZw... --window 7155690 --output /tmp/

    # Download latest N files for a hotkey
    python scripts/download_parquet.py --hotkey 5H1NNZw... --latest 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set


def get_s3_client():
    """Create boto3 S3 client for R2."""
    import boto3

    account_id = os.getenv("R2_ACCOUNT_ID")
    if not account_id:
        print("ERROR: R2_ACCOUNT_ID not set")
        sys.exit(1)

    # Try read credentials first, fall back to write
    access_key = os.getenv("R2_READ_ACCESS_KEY_ID") or os.getenv("R2_WRITE_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_READ_SECRET_ACCESS_KEY") or os.getenv("R2_WRITE_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        print("ERROR: R2 credentials not set")
        print("Need R2_READ_ACCESS_KEY_ID/R2_READ_SECRET_ACCESS_KEY or R2_WRITE_ACCESS_KEY_ID/R2_WRITE_SECRET_ACCESS_KEY")
        sys.exit(1)

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def get_bucket_name() -> str:
    """Get bucket name from environment."""
    bucket = os.getenv("R2_BUCKET_ID")
    if not bucket:
        print("ERROR: R2_BUCKET_ID not set")
        sys.exit(1)
    return bucket


def list_parquet_files(prefix: str = "", limit: int = 100) -> list[dict]:
    """List parquet files in bucket."""
    client = get_s3_client()
    bucket = get_bucket_name()

    files = []
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append({
                    "key": key,
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                })
            if len(files) >= limit:
                break
        if len(files) >= limit:
            break

    # Sort by last modified (newest first)
    files.sort(key=lambda x: x["last_modified"], reverse=True)
    return files


def download_file(key: str, output_path: str) -> bool:
    """Download a file from R2."""
    client = get_s3_client()
    bucket = get_bucket_name()

    try:
        print(f"Downloading: {key}")
        client.download_file(bucket, key, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved to: {output_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download parquet files from R2")
    parser.add_argument("--list", action="store_true", help="List parquet files in bucket")
    parser.add_argument("--hotkey", help="Miner hotkey to download")
    parser.add_argument("--window", type=int, help="Window number to download")
    parser.add_argument("--latest", type=int, help="Download latest N files for hotkey")
    parser.add_argument("--output", "-o", default=".", help="Output directory (default: current)")
    parser.add_argument("--limit", type=int, default=50, help="Max files to list (default: 50)")

    args = parser.parse_args()

    if args.list:
        print(f"Listing parquet files in bucket: {get_bucket_name()}")
        prefix = ""
        if args.hotkey:
            prefix = args.hotkey[:20]  # Partial prefix match

        files = list_parquet_files(prefix=prefix, limit=args.limit)

        if not files:
            print("No parquet files found")
            return

        print(f"\nFound {len(files)} parquet files:\n")
        for f in files:
            size_mb = f["size"] / (1024 * 1024)
            print(f"  {f['key']}")
            print(f"    Size: {size_mb:.2f} MB | Modified: {f['last_modified']}")
        return

    if args.hotkey and args.window:
        # Download specific file
        key = f"{args.hotkey}-window-{args.window}.parquet"
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / key

        if download_file(key, str(output_path)):
            print(f"\nDownload complete!")
        return

    if args.hotkey and args.latest:
        # Download latest N files for hotkey
        print(f"Finding latest {args.latest} files for hotkey: {args.hotkey[:20]}...")
        files = list_parquet_files(prefix=args.hotkey[:20], limit=args.latest)

        if not files:
            print("No files found")
            return

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for f in files[:args.latest]:
            key = f["key"]
            output_path = output_dir / key
            if download_file(key, str(output_path)):
                downloaded += 1

        print(f"\nDownloaded {downloaded}/{args.latest} files")
        return

    # No valid action
    parser.print_help()


if __name__ == "__main__":
    main()
