"""Download a public-domain/MIT-compatible 52-card face demo set.

The images come from:
https://github.com/hayeah/playing-cards-assets

The repository states that the original card artwork is public domain and
ships the processed asset repository under the MIT license.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master"
RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]
CARD_FILES = [f"png/{rank}_of_{suit}.png" for suit in SUITS for rank in RANKS]
SAMPLE_FILES = [
    "png/ace_of_spades.png",
    "png/10_of_diamonds.png",
    "png/queen_of_hearts.png",
    "png/king_of_clubs.png",
]


def download(url: str, destination: Path, timeout_seconds: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "E-Puzzle-Vision-Demo/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    destination.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("assets") / "card_templates"),
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Download only A-spades, 10-diamonds, Q-hearts, and K-clubs",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [*(SAMPLE_FILES if args.sample_only else CARD_FILES), "LICENSE"]

    def fetch(relative_path: str) -> tuple[str, str]:
        url = f"{BASE_URL}/{relative_path}"
        filename = Path(relative_path).name
        destination = output_dir / filename
        if destination.exists() and not args.force:
            return filename, "exists"
        download(url, destination, args.timeout)
        return filename, "downloaded"

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(fetch, relative): relative for relative in files}
        for future in as_completed(futures):
            filename, status = future.result()
            completed += 1
            print(f"[{completed:02d}/{len(files):02d}] {status}: {filename}")

    card_count = len(list(output_dir.glob("*_of_*.png")))
    print(f"card templates available: {card_count}")
    if not args.sample_only and card_count != 52:
        raise RuntimeError(f"Expected 52 card faces, found {card_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
