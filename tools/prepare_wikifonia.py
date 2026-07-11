#!/usr/bin/env python3
# prepare_wikifonia.py — Wikifonia ダンプ（リードシート約6,400曲 .mxl）をコーパスに変換。
#
#   入力: tools/data/raw/wikifonia-mxl/**/*.mxl（fetch_data.sh が Wayback Machine から取得）
#   出力: tools/data/wikifonia.corpus        （メロディ+コード 8小節チャンク → メロディTF用）
#         tools/data/wikifonia_chords.corpus （フル曲コード進行 → コードTF用）
#
#   ・解析は prepare_genres.py の music21 パスを再利用（4/4のみ・キー正規化・保持トークン形式）
#   ・ジャンルメタデータが無いため genre は "songbook" 固定。既存ジャンル(pop等)の分布を
#     汚さず、生成時は 'pop' 指定なので songbook の様式が直接出ることもない
#     （メロディTFの重み共有によるデータ増の恩恵だけを得る）。
#   ・OpenEWLD は Wikifonia の部分集合のため重複が出るが、完全一致は train.py 読込時に除去される。
#   使い方:  python3 tools/prepare_wikifonia.py [--limit N]
#   ※ 全量パースは music21 で数時間かかる。

import argparse
import glob
import os
import sys

from music21 import converter

import prepare_genres as PG

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "wikifonia-mxl")
OUT_MEL = os.path.join(HERE, "data", "wikifonia.corpus")
OUT_CH = os.path.join(HERE, "data", "wikifonia_chords.corpus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="処理する曲数の上限(0=全部)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    files = sorted(glob.glob(os.path.join(RAW, "**", "*.mxl"), recursive=True))
    if not files:
        print(f"入力が見つかりません（{RAW}）。先に tools/fetch_data.sh を実行してください。", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        files = files[:args.limit]
    print(f"Wikifonia: {len(files)} ファイルを解析", flush=True)

    stats = {}
    n_ok = n_skip = 0
    with open(OUT_MEL, "w", encoding="utf-8") as out_mel, open(OUT_CH, "w", encoding="utf-8") as out_ch:
        out_mel.write("# 自動生成（prepare_wikifonia.py）: Wikifonia のメロディ＋コード（8小節チャンク）。\n\n")
        out_ch.write("# 自動生成（prepare_wikifonia.py）: Wikifonia のフル曲コード進行。\n\n")
        for i, fp in enumerate(files):
            title = os.path.splitext(os.path.basename(fp))[0]
            try:
                s = converter.parse(fp)
                res = PG.extract_leadsheet(s)
                if res:
                    PG.emit(res[0], "songbook", title, out_mel, out_ch, stats)
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception:
                n_skip += 1
            if (i + 1) % 500 == 0:
                print(f"  ...{i+1}/{len(files)}  採用 {n_ok} / スキップ {n_skip}", flush=True)
    print(f"完了: {n_ok} 曲を採用（スキップ {n_skip}: 4/4以外・コード無し・解析失敗）")
    print(f"→ {os.path.relpath(OUT_MEL, HERE)} / {os.path.relpath(OUT_CH, HERE)}")


if __name__ == "__main__":
    main()
