#!/usr/bin/env python3
# validate_corpus.py — コーパスの品質検査。前処理を変えたら必ず実行する。
#
# 検査項目:
#   ・コードトークンがローマ数字として解釈可能か（語彙の汚れ検出）
#   ・メロディ: 各小節の拍数合計が4か / 音名トークンが正しいか
#   ・メロディ付きエントリ: コード拍数 == 小節数×4 か
#   ・chords の先頭が "_"（保持）で始まっていないか
#   ・完全重複エントリ（コード+メロディが同一）の検出
#   ・統計: 保持トークン率 / スラッシュ率 / 彩り率
#   使い方:  python3 tools/validate_corpus.py

import glob
import os
import re
import sys
from collections import Counter

import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROMAN_RE = re.compile(r"^[b#]{0,2}(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)"
                      r"[A-Za-z0-9°ø＋+]*(/(1[01]|\d))?$")
NOTE_RE = re.compile(r"^(R|[A-G](##|bb|#|b)?-?\d+):\d+(\.\d+)?$")


def check_file(path):
    songs = T.parse_corpus(path)
    errs = Counter()
    n_hold = n_tok = n_slash = n_ev = 0
    seen = set()
    dup = 0
    for s in songs:
        ch = s["chords"]
        if not ch:
            errs["chords空"] += 1
            continue
        if ch[0] == "_":
            errs["先頭が保持(_)"] += 1
        for t in ch:
            n_tok += 1
            if t == "_":
                n_hold += 1
                continue
            n_ev += 1
            if "/" in t:
                n_slash += 1
            if not ROMAN_RE.match(t):
                errs[f"不正コード: {t}"] += 1
        bars = s.get("melody")
        if bars:
            if len(ch) != 4 * len(bars):
                errs["コード拍数≠小節数×4"] += 1
            for bar in bars:
                total = 0.0
                for tok in bar:
                    if not NOTE_RE.match(tok):
                        errs[f"不正メロディ: {tok}"] += 1
                        continue
                    total += float(tok.split(":")[1])
                if abs(total - 4.0) > 1e-6:
                    errs[f"小節拍数={total:g}"] += 1
        key = (" ".join(ch), "|".join(" ".join(b) for b in (bars or [])))
        if key in seen:
            dup += 1
        seen.add(key)
    name = os.path.basename(path)
    stat = (f"{len(songs):>6}曲  保持率={n_hold/max(1,n_tok)*100:4.1f}%  "
            f"スラッシュ={n_slash/max(1,n_ev)*100:4.1f}%  重複={dup}")
    print(f"{name:26s} {stat}")
    if errs:
        for k, v in errs.most_common(8):
            print(f"    ! {k} ×{v}")
    return sum(errs.values())


def main():
    total = 0
    for p in [os.path.join(HERE, "corpus.txt")] + sorted(glob.glob(os.path.join(HERE, "data", "*.corpus"))):
        total += check_file(p)
    print("---")
    print("問題なし" if total == 0 else f"要確認: {total} 件")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
