#!/usr/bin/env python3
# prepare_pop909.py — POP909 を「メロディ＋コード」コーパス(tools/data/pop909.corpus)に変換。
#
#   入力: tools/data/raw/POP909-Dataset/POP909/<id>/ 以下
#          - <id>.mid          (MELODY / BRIDGE / PIANO トラック)
#          - chord_midi.txt     (開始秒 終了秒 コード)
#          - beat_midi.txt      (時刻秒 拍 ダウンビート)  ← 3列目=1 が小節頭
#          - key_audio.txt      (開始秒 終了秒 キー 例 'Gb:maj')
#   出力: tools/data/pop909.corpus
#
#   4/4（1小節=4拍）の小節のみ採用。キーは平行長調に正規化し、メロディは音名で書き出す
#   （train.py 側で主音相対に正規化される）。データ量を増やすため:
#     ・MELODY だけでなく BRIDGE（サブメロディ）も抽出
#     ・1曲を CHUNK 小節ごとに分割して複数系列にする（フレーズ単位の学習に有効）
#   要 pretty_midi。
#   使い方:  python3 tools/prepare_pop909.py [--limit N] [--chunk 8] [--max-bars 64]

import os, sys, glob, argparse
import chordlib as cl

try:
    import pretty_midi
except ImportError:
    print("pretty_midi が必要です:  pip install pretty_midi", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
POP = os.path.join(HERE, "data", "raw", "POP909-Dataset", "POP909")
OUT = os.path.join(HERE, "data", "pop909.corpus")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEYNAME = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]   # 16分(0.25)・付点8分(0.75)まで表現
MIN_BARS = 4
TRACKS = ("MELODY", "BRIDGE")


def midi_to_name(m):
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def quant_dur(x, remaining):
    x = min(x, remaining)
    cand = [d for d in DURS if d <= remaining + 1e-6] or [0.5]
    return min(cand, key=lambda d: abs(d - x))


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                rows.append(p)
    return rows


def load_beats(path):
    out = []
    for p in read_rows(path):
        try:
            t = float(p[0]); db = float(p[-1])
        except ValueError:
            continue
        out.append((t, db >= 0.5))
    out.sort()
    return out


def load_chords(path):
    out = []
    for p in read_rows(path):
        try:
            s, e = float(p[0]), float(p[1])
        except ValueError:
            continue
        out.append((s, e, " ".join(p[2:])))
    return out


def chord_at(chords, t0, t1):
    best, best_ov = None, 0.0
    for s, e, lab in chords:
        ov = min(e, t1) - max(s, t0)
        if ov > best_ov:
            best_ov, best = ov, lab
    return best


def get_track_notes(pm, name):
    inst = next((i for i in pm.instruments if (i.name or "").strip().upper() == name), None)
    if inst is None:
        return None
    notes = sorted(inst.notes, key=lambda n: (n.start, -n.pitch))
    mono, last_end = [], -1.0
    for n in notes:                                    # 単旋律化：重なりは先の音を優先
        if n.start < last_end - 1e-3:
            continue
        mono.append(n); last_end = n.end
    return mono


def measures_of_track(notes, downbeats, beats, chords, tonic):
    """1トラックを小節列に。返り値: [(beat_romans[4], bar_str)]。
       beat_romans = 拍ごとのコード（無い拍は直前を保持）。コード無し/音無しの小節は捨てる。"""
    tonic_ref = 60 + tonic
    out = []
    for i in range(len(downbeats) - 1):
        m0, m1 = downbeats[i], downbeats[i + 1]
        nbeats = sum(1 for (t, _) in beats if m0 - 1e-6 <= t < m1 - 1e-6)
        if nbeats != 4:
            continue
        beat_dur = (m1 - m0) / 4.0
        # 拍ごとのコード（無い拍は直前を保持）
        beat_romans, prev_r = [], None
        for k in range(4):
            lab = chord_at(chords, m0 + k * beat_dur, m0 + (k + 1) * beat_dur)
            parsed = cl.parse_chord_label(lab) if lab else None
            r = cl.chord_to_roman(parsed[0], tonic, parsed[1], parsed[2]) if parsed else prev_r
            beat_romans.append(r)
            if r is not None:
                prev_r = r
        firstnn = next((r for r in beat_romans if r is not None), None)
        if firstnn is None:
            continue                                     # 小節にコードが全く無い
        beat_romans = [r if r is not None else firstnn for r in beat_romans]

        bar_notes = [n for n in notes if m0 - 1e-6 <= n.start < m1 - 1e-6]
        if not bar_notes:
            continue
        seq, cursor = [], 0.0
        for ni, n in enumerate(bar_notes):
            remaining = 4.0 - cursor
            if remaining <= 1e-6:
                break
            pos = (n.start - m0) / beat_dur
            gap = pos - cursor
            if gap >= 0.25:
                rd = quant_dur(gap, remaining)
                seq.append(("R", rd)); cursor += rd; remaining = 4.0 - cursor
                if remaining <= 1e-6:
                    break
            pitch = n.pitch
            while pitch - tonic_ref > 24:
                pitch -= 12
            while pitch - tonic_ref < -12:
                pitch += 12
            # レガート正規化: ボーカルMIDIは音符実長が発音間隔より短く「音+マイクロ休符」の
            # アーティファクトが大量に出る(トークンの12.7%)。次の発音までの隙間が半拍未満なら
            # 音を伸ばして隙間を吸収する（半拍以上の隙間は本物の休符として残す）。
            raw = (n.end - n.start) / beat_dur
            nxt = bar_notes[ni + 1].start if ni + 1 < len(bar_notes) else m1
            ioi = (nxt - n.start) / beat_dur
            if ioi - raw < 0.5:
                raw = ioi
            dur = quant_dur(raw, remaining)
            seq.append((midi_to_name(pitch), dur)); cursor += dur
        if cursor < 4.0 - 1e-6:
            seq.append(("R", round(4.0 - cursor, 3)))
        out.append((beat_romans, " ".join(f"{p}:{('%g' % d)}" for p, d in seq)))
    return out


def build_blocks(folder, chunk, max_bars):
    """1曲から (blocks, chord_prog) を返す。
       blocks = メロディ+コードの (key, chords, bars) 群（トラック×チャンク）。
       chord_prog = フル曲の小節単位コード進行（構造学習用。MELODY優先）。"""
    sid = os.path.basename(folder)
    key_rows = load_chords(os.path.join(folder, "key_audio.txt"))
    tonic = cl.key_tonic_pc(key_rows[0][2]) if key_rows else None
    if tonic is None:
        return [], None
    beats = load_beats(os.path.join(folder, "beat_midi.txt"))
    chords = load_chords(os.path.join(folder, "chord_midi.txt"))
    if not beats or not chords:
        return [], None
    downbeats = [t for (t, db) in beats if db]
    pm = pretty_midi.PrettyMIDI(os.path.join(folder, sid + ".mid"))

    blocks = []
    chord_prog, first_prog = None, None
    for tname in TRACKS:
        notes = get_track_notes(pm, tname)
        if not notes:
            continue
        meas = measures_of_track(notes, downbeats, beats, chords, tonic)[:max_bars]
        romans = [r for r, _ in meas]
        if first_prog is None and len(romans) >= MIN_BARS:
            first_prog = romans
        if tname == "MELODY" and len(romans) >= MIN_BARS:
            chord_prog = romans
        # CHUNK 小節ごとに分割
        for c in range(0, len(meas), chunk):
            seg = meas[c:c + chunk]
            if len(seg) < MIN_BARS:
                continue
            blocks.append((KEYNAME[tonic],
                           [r for r, _ in seg],
                           [b for _, b in seg]))
    return blocks, (chord_prog or first_prog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="使う曲数の上限(0=全部)")
    ap.add_argument("--chunk", type=int, default=8, help="1系列の小節数（分割単位）")
    ap.add_argument("--max-bars", type=int, default=64, help="1曲1トラックあたり最大小節数")
    args = ap.parse_args()

    folders = sorted(f for f in glob.glob(os.path.join(POP, "*")) if os.path.isdir(f))
    if not folders:
        print(f"POP909 が見つかりません（探索先: {POP}）。先に tools/fetch_data.sh を実行してください。", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        folders = folders[:args.limit]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    OUT_CHORDS = os.path.join(HERE, "data", "pop909_chords.corpus")
    n_blocks, n_songs, n_prog = 0, 0, 0
    with open(OUT, "w", encoding="utf-8") as out, open(OUT_CHORDS, "w", encoding="utf-8") as outc:
        out.write("# 自動生成（prepare_pop909.py）: POP909 のメロディ＋コード。MELODY/BRIDGE を8小節チャンクに分割。\n\n")
        outc.write("# 自動生成（prepare_pop909.py）: POP909 のフル曲コード進行（構造学習用・メロディなし）。\n\n")
        for i, folder in enumerate(folders):
            try:
                blocks, chord_prog = build_blocks(folder, args.chunk, args.max_bars)
            except Exception as ex:
                print(f"  ! skip {os.path.basename(folder)}: {ex}", file=sys.stderr)
                continue
            if blocks:
                n_songs += 1
            if chord_prog:                               # chords は保持トークン形式（1トークン=1拍, _=保持）
                outc.write(f"# POP909/{os.path.basename(folder)}\ntime: 4/4\nkey: C\nchords: "
                           + " ".join(cl.to_holds([r for br in chord_prog for r in br])) + "\n\n")
                n_prog += 1
            for j, (key, chords, bars) in enumerate(blocks):
                out.write(f"# POP909/{os.path.basename(folder)}#{j}\n")
                out.write(f"time: 4/4\nkey: {key}\n")
                out.write("chords: " + " ".join(cl.to_holds([r for br in chords for r in br])) + "\n")
                out.write("melody: " + " | ".join(bars) + "\n\n")
                n_blocks += 1
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1} 曲処理 / 系列 {n_blocks}")
    print(f"{n_songs} 曲から {n_blocks} 系列 → {os.path.relpath(OUT, HERE)}")
    print(f"{n_prog} 曲のフル進行 → {os.path.relpath(OUT_CHORDS, HERE)}")


if __name__ == "__main__":
    main()
