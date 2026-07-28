"""Fetch a pinned external input without adding it to version control."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/Duzhaozhen/OmniAge/"
    "77f8eb56ed7436ce93593142944362369a86ee6b/"
    "OmniAgeR/inst/extdata/omniager_lung_inv.rds"
)
SHA256 = "5a96d2f9f8f2807e220f9d699fc6f5364a80a49ccc946b0950357d13e36f2beb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/omniager_lung_inv.rds"),
    )
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != SHA256:
        raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {actual_hash}")
    arguments.output.write_bytes(content)
    print(f"verified external input at {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
