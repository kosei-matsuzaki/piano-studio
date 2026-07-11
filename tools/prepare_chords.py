#!/usr/bin/env python3
# prepare_chords.py — コード専用データセット(anime-song / tmc323)を
# 「コードのみコーパス」(tools/data/chords_only.corpus) に変換する。
#
#   入力: tools/data/raw/ 以下にクローンした
#          - Chord-Dataset/        (anime-song, chord/*.txt, "開始\t終了\tコード")
#          - Chord-Annotations/    (tmc323, RWC_Pop_Chords/*.lab, uspopLabels/*.lab)
#   出力: tools/data/chords_only.corpus  （key: C 固定＋保持トークン形式のコード列。メロディなし）
#
#   コードの長さ（拍数）は開始/終了秒から推定する:
#     1) 半拍グリッドの総当りで拍長（テンポ）を推定
#     2) 曲を進みながら拍長を適応更新（テンポドリフト追従）し、半拍数に量子化
#     3) 端数（食い気味のコードチェンジ等）は次のコードへ持ち越して整数拍に丸める
#     4) コードの85%以上が拍に整合した曲だけ実長を採用。失敗曲は従来どおり1コード=1小節
#   キーは付いていないので chordlib.estimate_tonic() でダイアトニック被覆から推定する。
#   使い方:  python3 tools/prepare_chords.py

import os, sys, glob
import chordlib as cl

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "chords_only.corpus")

MIN_CHORDS = 3
MAX_BEATS = 256          # 1曲の上限拍数（pop909 の --max-bars 64 と揃える）
MAX_HOLD = 16            # 1コードの上限拍数（イントロのドローン等の暴走防止）
MIN_DUR = 0.15           # これ未満の極短イベントはノイズとして無視（秒）
OK_RATE = 0.85           # 拍整合率がこれ以上なら実長を採用


def read_lab(path):
    """'開始 終了 ラベル'（空白 or タブ）の行を [(start,end,label)] に。ヘッダ行は無視。"""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace("\t", " ").split()
            if len(parts) < 3:
                continue
            try:
                s, e = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            rows.append((s, e, " ".join(parts[2:])))
    return rows


def parse_events(rows):
    """コード行 → [(root_pc, qual, bass_pc, 長さ秒)]。N/解釈不能/極短は捨てる。"""
    events = []
    for s, e, label in rows:
        if e - s < MIN_DUR:
            continue
        parsed = cl.parse_chord_label(label)
        if parsed is None:
            continue
        root_pc, qual, bass_pc = parsed
        events.append((root_pc, qual, bass_pc, e - s))
    return events


def fit_beat(durs):
    """半拍グリッドの総当り（約50〜200BPM）で拍長を推定。90パーセンタイル誤差が最小のものを返す。"""
    best_b, best_err = None, 1e9
    b = 0.30
    while b <= 1.20:
        half = b / 2
        errs = sorted(abs(d - max(1, round(d / half)) * half) / half for d in durs)
        err = errs[int(len(errs) * 0.9)]
        if err < best_err:
            best_b, best_err = b, err
        b += 0.004
    return best_b


def quantize_beats(durs):
    """各コードの長さ（秒）→ 拍数のリスト。返り値 (beats, 整合率)。
       半拍単位で量子化しつつ拍長を追従更新し、端数は次コードへ持ち越して整数拍にする。"""
    half = fit_beat(durs[:48]) / 2
    quant, nerr = [], 0
    for d in durs:
        k = max(1, round(d / half))
        if abs(d - k * half) / half < 0.3:
            half = 0.85 * half + 0.15 * (d / k)     # テンポドリフト追従
        else:
            nerr += 1
        quant.append(k / 2)                          # 拍数（0.5刻み）
    beats, carry = [], 0.0
    for q in quant:
        v = q + carry
        k = max(1, round(v))
        carry = v - k                                # 食い気味の変化を次の拍頭に吸収
        beats.append(min(k, MAX_HOLD))
    return beats, 1.0 - nerr / max(1, len(quant))


def song_to_holds(rows):
    """コードイベント行 → 保持トークン列（I _ _ _ IV _ ...）。失敗時 None。"""
    events = parse_events(rows)
    if len(events) < MIN_CHORDS:
        return None
    tonic = cl.estimate_tonic([(r, q, w) for r, q, b, w in events])
    beats, ok = quantize_beats([d for _, _, _, d in events])
    if ok < OK_RATE:
        beats = [4] * len(events)                    # フォールバック: 1コード=1小節
    # 拍中央値が2〜4拍に収まるよう全体スケールを補正（拍の取り違え対策）
    med = sorted(beats)[len(beats) // 2]
    if med >= 6:
        beats = [max(1, round(k / 2)) for k in beats]
    elif med <= 1:
        beats = [min(k * 2, MAX_HOLD) for k in beats]
    # ローマ数字化して、連続する同一コードは拍数を合算
    merged = []                                      # [(roman, beats)]
    for (root_pc, qual, bass_pc, _), k in zip(events, beats):
        r = cl.chord_to_roman(root_pc, tonic, qual, bass_pc)
        if merged and merged[-1][0] == r:
            merged[-1] = (r, min(merged[-1][1] + k, MAX_HOLD))
        else:
            merged.append((r, k))
    if len(merged) < MIN_CHORDS:
        return None
    toks = []
    for r, k in merged:
        if len(toks) + k > MAX_BEATS:
            break
        toks.append(r)
        toks.extend("_" * (k - 1))
    return toks if len(set(t for t in toks if t != "_")) >= MIN_CHORDS else None


def collect_files():
    files = []
    files += glob.glob(os.path.join(RAW, "Chord-Dataset", "**", "*.txt"), recursive=True)
    files += glob.glob(os.path.join(RAW, "Chord-Annotations", "**", "*.lab"), recursive=True)
    files += glob.glob(os.path.join(RAW, "McGill-Billboard", "**", "*.lab"), recursive=True)      # 米ポップス890曲
    files += glob.glob(os.path.join(RAW, "isophonics", "**", "chordlab", "**", "*.lab"), recursive=True)   # Beatles等225曲
    # index/README 等を除外
    return [f for f in files if not os.path.basename(f).lower().startswith(("readme", "index", "license"))]


def main():
    files = collect_files()
    if not files:
        print(f"入力が見つかりません。先に tools/fetch_data.sh を実行してください（探索先: {RAW}）", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_ok = 0
    with open(OUT, "w", encoding="utf-8") as out:
        out.write("# 自動生成（prepare_chords.py）: anime-song / tmc323 のコード進行（保持トークン形式・拍数は秒から推定）。メロディなし＝コードモデル専用。\n\n")
        for path in sorted(files):
            try:
                toks = song_to_holds(read_lab(path))
            except Exception as ex:
                print(f"  ! skip {os.path.basename(path)}: {ex}", file=sys.stderr)
                continue
            if not toks:
                continue
            out.write(f"# {os.path.relpath(path, RAW)}\n")
            out.write("time: 4/4\nkey: C\n")
            out.write("chords: " + " ".join(toks) + "\n\n")
            n_ok += 1
    print(f"{n_ok} 曲を書き出しました → {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
