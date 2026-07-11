#!/usr/bin/env python3
# prepare_pop909_dyn.py — POP909 の実演奏ベロシティから「小節ごとの強弱曲線」学習データを作る。
#
# 目的: 「この曲のここは盛り上げる/ここは優しく」を固定ルールでなくデータから学ばせる。
# 各小節の音楽的内容(コード・メロディの形) → その小節の平均ベロシティ(曲内z正規化) を対にする。
#
#   出力: tools/data/pop909_dyn.npz  (X: [N,24] 特徴, y: [N] z正規化ベロシティ, song: [N] 曲ID)
#   特徴ベクトル(neural.js dynamicsCurve と完全に一致させること):
#     [0..11] 拍コードのルートpcヒストグラム(主音相対, /4)
#     [12]    短三和音系(小文字ローマ数字)の割合
#     [13]    彩りコード(7/9/11/13/sus/aug/°/ø/6)の割合
#     [14]    メロディ音符密度 (音数/8, 上限2)
#     [15]    平均音高(主音相対オフセット/24)
#     [16]    最高音(オフセット/24)
#     [17]    休符割合(拍/4)
#     [18]    ロングノート(2拍以上)の有無
#     [19..22] フレーズ内小節位置 one-hot (小節番号%4)
#     [23]    曲内位置 (小節番号/32, 上限1)
#   使い方:  python3 tools/prepare_pop909_dyn.py

import os, sys, glob, re
import numpy as np

import chordlib as cl
import prepare_pop909 as PP

try:
    import pretty_midi
except ImportError:
    print("pretty_midi が必要です:  pip install pretty_midi", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pop909_dyn.npz")

RICH_RE = re.compile(r"7|9|11|13|sus|aug|°|ø|6")
NUM_RE = re.compile(r"^([b#]*)(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)")


def roman_root_pc(roman):
    """ローマ数字 → 主音相対の半音 (neural.js romanToChord と同じ規約)。"""
    s = str(roman).split("/")[0]
    acc = 0
    while s and s[0] in "b#":
        acc += 1 if s[0] == "#" else -1
        s = s[1:]
    m = NUM_RE.match(s)
    if not m:
        return 0
    deg = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}[m.group(2).upper()]
    return ([0, 2, 4, 5, 7, 9, 11][deg] + acc) % 12


def bar_features(beat_romans, notes, bar_index):
    """1小節の特徴ベクトル(24次元)。notes = [(off, dur_beats, vel)] (off=主音相対)。"""
    f = np.zeros(24, dtype=np.float64)
    for r in beat_romans:
        f[roman_root_pc(r)] += 0.25
        base = str(r).split("/")[0]
        m = NUM_RE.match(base.lstrip("b#"))
        if m and m.group(2).islower():
            f[12] += 0.25
        if RICH_RE.search(base):
            f[13] += 0.25
    pitched = [(o, d) for o, d, v in notes]
    if pitched:
        f[14] = min(2.0, len(pitched) / 8.0)
        f[15] = np.mean([o for o, _ in pitched]) / 24.0
        f[16] = max(o for o, _ in pitched) / 24.0
        f[18] = 1.0 if any(d >= 2.0 for _, d in pitched) else 0.0
    dur_sum = sum(d for _, d in pitched)
    f[17] = max(0.0, min(1.0, (4.0 - dur_sum) / 4.0))
    f[19 + (bar_index % 4)] = 1.0
    f[23] = min(1.0, bar_index / 32.0)
    return f


def song_bars(folder):
    """1曲 → [(features, mean_vel or None)]。prepare_pop909 と同じ小節切り・キー正規化。"""
    sid = os.path.basename(folder)
    key_rows = PP.load_chords(os.path.join(folder, "key_audio.txt"))
    tonic = cl.key_tonic_pc(key_rows[0][2]) if key_rows else None
    if tonic is None:
        return []
    beats = PP.load_beats(os.path.join(folder, "beat_midi.txt"))
    chords = PP.load_chords(os.path.join(folder, "chord_midi.txt"))
    if not beats or not chords:
        return []
    downbeats = [t for (t, db) in beats if db]
    pm = pretty_midi.PrettyMIDI(os.path.join(folder, sid + ".mid"))
    mel = PP.get_track_notes(pm, "MELODY")
    if not mel:
        return []
    tonic_ref = 60 + tonic
    out = []
    bar_index = 0
    for i in range(len(downbeats) - 1):
        m0, m1 = downbeats[i], downbeats[i + 1]
        nbeats = sum(1 for (t, _) in beats if m0 - 1e-6 <= t < m1 - 1e-6)
        if nbeats != 4:
            continue
        beat_dur = (m1 - m0) / 4.0
        beat_romans, prev_r = [], None
        for k in range(4):
            lab = PP.chord_at(chords, m0 + k * beat_dur, m0 + (k + 1) * beat_dur)
            parsed = cl.parse_chord_label(lab) if lab else None
            r = cl.chord_to_roman(parsed[0], tonic, parsed[1], parsed[2]) if parsed else prev_r
            beat_romans.append(r)
            if r is not None:
                prev_r = r
        firstnn = next((r for r in beat_romans if r is not None), None)
        if firstnn is None:
            continue
        beat_romans = [r if r is not None else firstnn for r in beat_romans]
        notes = []
        for n in mel:
            if not (m0 - 1e-6 <= n.start < m1 - 1e-6):
                continue
            pitch = n.pitch
            while pitch - tonic_ref > 24:
                pitch -= 12
            while pitch - tonic_ref < -12:
                pitch += 12
            notes.append((pitch - tonic_ref, (n.end - n.start) / beat_dur, n.velocity))
        vel = float(np.mean([v for _, _, v in notes])) if notes else None
        out.append((bar_features(beat_romans, notes, bar_index), vel))
        bar_index += 1
    return out


def main():
    folders = sorted(f for f in glob.glob(os.path.join(PP.POP, "*")) if os.path.isdir(f))
    if not folders:
        print("POP909 が見つかりません。先に tools/fetch_data.sh を実行してください。", file=sys.stderr)
        sys.exit(1)
    X, y, song = [], [], []
    n_songs = 0
    for si, folder in enumerate(folders):
        try:
            bars = song_bars(folder)
        except Exception as ex:
            print(f"  ! skip {os.path.basename(folder)}: {ex}", file=sys.stderr)
            continue
        vels = [v for _, v in bars if v is not None]
        if len(vels) < 16:
            continue
        mu, sd = np.mean(vels), max(1.0, np.std(vels))    # 曲内z正規化(録音ごとの音量差を除去)
        n_songs += 1
        for feat, v in bars:
            X.append(feat)
            y.append((v - mu) / sd if v is not None else np.nan)   # 無音小節は NaN → 学習時マスク
            song.append(n_songs - 1)
        if (si + 1) % 200 == 0:
            print(f"  ...{si+1} 曲処理")
    X = np.asarray(X); y = np.asarray(y); song = np.asarray(song)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, song=song)
    valid = np.isfinite(y)
    print(f"{n_songs} 曲 / {len(y)} 小節（有効 {valid.sum()}）→ {os.path.relpath(OUT, HERE)}")
    print(f"z分布: mean={np.nanmean(y):.2f} std={np.nanstd(y):.2f}")


if __name__ == "__main__":
    main()
