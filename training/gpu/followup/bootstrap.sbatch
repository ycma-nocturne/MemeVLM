#!/bin/bash -l
set -euo pipefail
umask 077

root=$(cd -- "${1:?provide the scoped project directory}" && pwd -P)
work=$(cd -- "${2:?provide the follow-up directory}" && pwd -P)
case "$work" in
    "$root"/*) ;;
    *) printf 'The follow-up directory must be inside the project directory.\n' >&2; exit 2 ;;
esac
cd "$root"
test -s project/training/gpu/requirements.txt
test -s project/training/gpu/launch.py
export TMPDIR="$work/tmp/${SLURM_JOB_ID:?run within a Slurm allocation}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$work/cache" XDG_CONFIG_HOME="$work/config"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv" UV_PYTHON_INSTALL_DIR="$root/env/python"
export UV_PYTHON_BIN_DIR="$root/env/bin" UV_NO_MODIFY_PATH=1 UV_LINK_MODE=copy
export HF_HOME="$XDG_CACHE_HOME/hf" HF_DATASETS_CACHE="$XDG_CACHE_HOME/hf/datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$root/env/bin" "$work/logs"
exec 9>"$root/.followup-bootstrap.lock"
flock -n 9

uv="$root/env/bin/uv"
if ! test -x "$uv"; then
    python3 -B - "$uv" <<'PY'
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

with urlopen("https://pypi.org/pypi/uv/0.9.13/json", timeout=45) as response:
    package = json.load(response)
wheel, = (item for item in package["urls"]
          if item["filename"].endswith("manylinux_2_17_x86_64.manylinux2014_x86_64.whl"))
with urlopen(wheel["url"], timeout=45) as response:
    payload = response.read()
if hashlib.sha256(payload).hexdigest() != wheel["digests"]["sha256"]:
    raise ValueError("uv wheel digest mismatch")
dest = Path(sys.argv[1])
tmp = dest.with_suffix(".pending")
with ZipFile(io.BytesIO(payload)) as archive:
    member, = (name for name in archive.namelist() if name.endswith(".data/scripts/uv"))
    tmp.write_bytes(archive.read(member))
tmp.chmod(0o700)
tmp.replace(dest)
PY
fi

if ! test -x "$root/.venv-cuda/bin/python"; then
    "$uv" venv --python 3.12.12 --managed-python "$root/.venv-cuda"
fi
python="$root/.venv-cuda/bin/python"

"$python" -B - <<'PY' > "$work/torch-wheels.txt"
from html.parser import HTMLParser
import re
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen

class Links(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.items = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "a" and (href := dict(attrs).get("href")):
            self.items.append(href)

for name, version in {"torch": "2.8.0", "torchvision": "0.23.0", "torchaudio": "2.8.0"}.items():
    index = f"https://download.pytorch.org/whl/cu128/{name}/"
    with urlopen(index, timeout=45) as response:
        links = Links(response.read().decode()).items
    match = set()
    for link in links:
        candidate = urljoin(index, link)
        parts = urlsplit(candidate)
        filename = unquote(parts.path.rsplit("/", 1)[-1])
        if (parts.scheme == "https" and parts.hostname in {"download.pytorch.org", "download-r2.pytorch.org"}
                and filename.startswith(f"{name}-{version}+cu128-cp312-cp312-")
                and filename.endswith("x86_64.whl")
                and re.fullmatch(r"sha256=[0-9a-f]{64}", parts.fragment)):
            match.add(candidate)
    wheel, = sorted(match)
    print(wheel)
PY
mapfile -t wheels < "$work/torch-wheels.txt"
test "${#wheels[@]}" -eq 3
"$uv" pip install --python "$python" --index-url https://pypi.org/simple \
    "${wheels[@]}" -r "$root/project/training/gpu/requirements.txt"
"$uv" pip check --python "$python"
"$uv" pip freeze --python "$python" > "$work/runtime-dependencies.txt"
PYTHONPATH="$root/project" "$python" -B - "$root/model" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
from training.gpu.launch import MODEL_FILES, MODEL_ID, MODEL_REVISION

snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=Path(sys.argv[1]),
                  allow_patterns=list(MODEL_FILES), max_workers=3, token=False)
PY
"$python" -B "$root/project/training/gpu/launch.py" verify-model --model "$root/model" > "$work/model.json"
