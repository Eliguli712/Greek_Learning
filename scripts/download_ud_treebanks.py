"""Download official UD Ancient Greek train/dev/test CoNLL-U files."""

from __future__ import annotations

from pathlib import Path
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "data" / "ud_treebanks"

UD_URLS = {
    "grc_proiel-ud-train.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master/grc_proiel-ud-train.conllu",
    "grc_proiel-ud-dev.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master/grc_proiel-ud-dev.conllu",
    "grc_proiel-ud-test.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master/grc_proiel-ud-test.conllu",
    "grc_perseus-ud-train.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu",
    "grc_perseus-ud-dev.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-dev.conllu",
    "grc_perseus-ud-test.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-test.conllu",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in UD_URLS.items():
        target = TARGET_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}")
            continue
        print(f"download: {url}")
        urllib.request.urlretrieve(url, target)
        text = target.read_text(encoding="utf-8")
        target.write_text(text.rstrip("\r\n") + "\n", encoding="utf-8")
        print(f"wrote: {target}")


if __name__ == "__main__":
    main()
