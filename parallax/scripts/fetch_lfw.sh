#!/usr/bin/env bash
# Fetch Labeled Faces in the Wild (original, UNALIGNED crops) + official pair lists.
#
# We deliberately take the raw `lfw` release, not `lfw-funneled` / `lfw-deepfunneled`:
# those were aligned by someone else's algorithm, and alignment is part of what this
# system has to figure out for itself.
#
# Source: figshare mirror maintained for scikit-learn (the UMass host is offline).
# Usage: scripts/fetch_lfw.sh [DEST_DIR]      (default: <project>/data)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/../data}"
mkdir -p "$DEST/dl" "$DEST/protocol"
DEST="$(cd "$DEST" && pwd)"

# name | figshare file id | sha256
FILES=(
  "lfw.tgz|5976018|055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
  "pairs.txt|5976006|ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
  "pairsDevTrain.txt|5976012|1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa"
  "pairsDevTest.txt|5976009|7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c"
)

fetch() {
  local name="$1" id="$2" sum="$3" out="$DEST/dl/$1"
  if [[ -f "$out" ]] && echo "$sum  $out" | sha256sum -c --status 2>/dev/null; then
    echo "ok (cached)  $name"; return
  fi
  local try
  for try in 1 2 3 4; do
    if curl -fL --retry 3 --retry-delay 2 -C - -o "$out" \
         "https://ndownloader.figshare.com/files/$id"; then break; fi
    echo "retry $try for $name" >&2; sleep $((try * 2))
  done
  echo "$sum  $out" | sha256sum -c --status || { echo "CHECKSUM MISMATCH: $name" >&2; exit 1; }
  echo "ok           $name"
}

for f in "${FILES[@]}"; do IFS='|' read -r n i s <<<"$f"; fetch "$n" "$i" "$s"; done

cp "$DEST/dl/pairs.txt" "$DEST/dl/pairsDevTrain.txt" "$DEST/dl/pairsDevTest.txt" "$DEST/protocol/"

# Archive is untrusted input: unpack into its own fresh directory, never over existing files.
if [[ ! -d "$DEST/lfw" ]]; then
  tmp="$(mktemp -d "$DEST/.unpack.XXXXXX")"
  tar -xzf "$DEST/dl/lfw.tgz" -C "$tmp" --no-same-owner --no-same-permissions
  mv "$tmp/lfw" "$DEST/lfw"
  rmdir "$tmp"
fi

nimg=$(find "$DEST/lfw" -name '*.jpg' | wc -l)
nid=$(find "$DEST/lfw" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "LFW ready: $nimg images, $nid identities -> $DEST/lfw"
[[ "$nimg" -eq 13233 && "$nid" -eq 5749 ]] || { echo "unexpected counts (want 13233 / 5749)" >&2; exit 1; }
