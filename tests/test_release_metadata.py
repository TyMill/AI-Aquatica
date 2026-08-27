import configparser
import json
import re
from pathlib import Path

import ai_aquatica

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_metadata_are_synchronized():
    config = configparser.ConfigParser()
    config.read(ROOT / "setup.cfg")
    setup_version = config["metadata"]["version"]

    cff_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cff_match = re.search(r'^version:\s*["\']?([^"\'\n]+)', cff_text, flags=re.MULTILINE)
    assert cff_match is not None
    cff_version = cff_match.group(1).strip()

    codemeta_version = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))["version"]

    assert setup_version == ai_aquatica.__version__ == cff_version == codemeta_version == "2.3.0"


def test_real_dataset_license_is_packaging_visible():
    assert (ROOT / "DATA_LICENSE.md").exists()
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include DATA_LICENSE.md" in manifest
    assert "recursive-include examples *.csv" in manifest
    assert (ROOT / "examples" / "data" / "water_quality.csv").exists()
