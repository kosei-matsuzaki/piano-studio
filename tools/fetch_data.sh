#!/usr/bin/env bash
# fetch_data.sh — 学習用データセットを tools/data/raw/ に取得する。
# ネット接続が必要。POP909 は数百MB程度。既に取得済みならスキップされる。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$HERE/data/raw"
mkdir -p "$RAW"
cd "$RAW"

clone() {  # clone <url> <dir>
  if [ -d "$2/.git" ] || [ -d "$2" ]; then
    echo "既に存在: $2 (スキップ)"
  else
    echo "clone: $1"
    git clone --depth 1 "$1" "$2"
  fi
}

# メロディ＋コード（後段メロディモデル用）
clone https://github.com/music-x-lab/POP909-Dataset.git   POP909-Dataset
# コード進行（前段コードモデル用）
clone https://github.com/anime-song/Chord-Dataset.git      Chord-Dataset
clone https://github.com/tmc323/Chord-Annotations.git      Chord-Annotations
# 多ジャンル（ジャンル条件づけ用）: OpenEWLD(jazz/rock/pop/folk/classical) と Nottingham(folk)
clone https://github.com/00sapo/OpenEWLD.git               OpenEWLD
clone https://github.com/jukedeck/nottingham-dataset.git   nottingham
# ※ classical(JSBコラール) は music21 内蔵のため取得不要

# McGill Billboard（米ポップス890曲のコード注釈 .lab、mirdata公式ミラー）
if [ ! -d McGill-Billboard ]; then
  echo "fetch: McGill Billboard"
  curl -sL -o billboard-lab.tar.gz "https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.gz?dl=1"
  mkdir -p bb-tmp && tar -xJf billboard-lab.tar.gz -C bb-tmp && mv bb-tmp/McGill-Billboard . && rm -rf bb-tmp billboard-lab.tar.gz
fi

# isophonics（Beatles/Queen/Carole King/Zweieck のコード注釈 .lab）
if [ ! -d isophonics ]; then
  echo "fetch: isophonics"
  mkdir -p isophonics
  for a in "The%20Beatles%20Annotations" "Queen%20Annotations" "Carole%20King%20Annotations" "Zweieck%20Annotations"; do
    curl -sL -A "Mozilla/5.0" -o iso.tar.gz "http://isophonics.net/files/annotations/${a}.tar.gz" && tar xzf iso.tar.gz -C isophonics
  done
  rm -f iso.tar.gz
fi

# Wikifonia（リードシート約6,400曲 .mxl。配布元消失のため Wayback Machine のスナップショット）
if [ ! -d wikifonia-mxl ]; then
  echo "fetch: Wikifonia (Wayback Machine)"
  curl -sL -o Wikifonia.zip "https://web.archive.org/web/20140701055029id_/http://www.synthzone.com/files/Wikifonia/Wikifonia.zip"
  mkdir -p wikifonia-mxl && unzip -q -o Wikifonia.zip -d wikifonia-mxl && rm Wikifonia.zip
fi

echo ""
echo "取得完了: $RAW"
echo "次の手順は docs/data-pipeline.md を参照。概要:"
echo "  pip install pretty_midi music21 torch numpy"
echo "  python3 tools/prepare_pop909.py && python3 tools/prepare_chords.py && python3 tools/prepare_genres.py && python3 tools/prepare_pop909_dyn.py"
echo "  python3 tools/train_chord_tf.py --epochs 90 && python3 tools/train_mel_tf.py --epochs 40 && python3 tools/train_dyn.py"
