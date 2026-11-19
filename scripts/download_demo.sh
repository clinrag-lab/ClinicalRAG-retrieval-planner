#!/usr/bin/env bash
# Download the open-access MIMIC-IV Clinical Database Demo v2.2 from PhysioNet.
# No credentials are required. Files are placed in data/mimic-iv-demo/ (git-ignored).
#
# Usage: bash scripts/download_demo.sh
set -euo pipefail

URL="https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/mimic-iv-demo"

if [ -d "$DEST/hosp" ]; then
    echo "Demo already present at $DEST; delete it to re-download."
    exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading MIMIC-IV demo v2.2..."
curl -fL --retry 3 -o "$TMP_DIR/demo.zip" "$URL"

echo "Extracting..."
unzip -q "$TMP_DIR/demo.zip" -d "$TMP_DIR/extract"

# The archive has a single top-level folder; move its contents into DEST.
SRC="$(find "$TMP_DIR/extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
mkdir -p "$DEST"
cp -R "$SRC"/. "$DEST"/

if [ -f "$DEST/SHA256SUMS.txt" ] && command -v sha256sum >/dev/null 2>&1; then
    echo "Verifying checksums..."
    (cd "$DEST" && sha256sum --quiet -c SHA256SUMS.txt)
fi

echo "Done. Demo data is in $DEST"
