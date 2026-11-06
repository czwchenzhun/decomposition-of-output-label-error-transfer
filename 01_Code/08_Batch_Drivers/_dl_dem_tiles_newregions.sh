#!/bin/bash
# Download missing Copernicus GLO-30 DEM tiles for the two new regions
#   R4_longzhong 103-106E / 34-37N (after expansion 33.6-37.4N, 102.6-106.4E) -> N33..N37 x E102..E106
#   R5_sichuan   105-108E / 29-32N (after expansion 28.6-32.4N, 104.6-108.4E) -> N28..N32 x E104..E108
# Source: AWS Open Data copernicus-dem-30m (public, no authentication required)
set -u
DEST="${RAW_DIR}/dem/DEM_tiles"
cd "$DEST" || { echo "DEST NOT FOUND: $DEST"; exit 1; }
echo "DEST=$DEST"

: > /tmp/tiles_all.txt
for N in 33 34 35 36 37; do
  for E in 102 103 104 105 106; do
    printf 'Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM.tif\n' "$N" "$E" >> /tmp/tiles_all.txt
  done
done
for N in 28 29 30 31 32; do
  for E in 104 105 106 107 108; do
    printf 'Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM.tif\n' "$N" "$E" >> /tmp/tiles_all.txt
  done
done
echo "manifest: $(wc -l < /tmp/tiles_all.txt) tiles"

: > /tmp/tiles_miss.txt
while read -r f; do
  [ -f "$f" ] || printf '%s\n' "$f" >> /tmp/tiles_miss.txt
done < /tmp/tiles_all.txt
echo "to download: $(wc -l < /tmp/tiles_miss.txt) tiles"

cat /tmp/tiles_miss.txt | xargs -P 5 -I{} sh -c '
  f="{}"; d="${f%.tif}"
  if curl -sSL --retry 2 --max-time 300 -o "$f.part" \
      "https://copernicus-dem-30m.s3.amazonaws.com/$d/$f"; then
    mv "$f.part" "$f"; echo "ok   $f"
  else
    rm -f "$f.part"; echo "FAIL $f"
  fi'

echo "========== DEM DOWNLOAD DONE =========="
grep -c "Copernicus" <(ls) || true
