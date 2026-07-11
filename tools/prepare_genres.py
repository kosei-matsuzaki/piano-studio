#!/usr/bin/env python3
# prepare_genres.py — 多ジャンルのリードシート/コラールを music21 で解析し、
# ジャンル付きコーパスに変換する（コード＋メロディをジャンル条件つき学習するため）。
#
#   ソース（tools/data/raw/ にクローン済み想定）:
#     - OpenEWLD/         : 502リードシート(.mxl) ＋ OpenEWLD.db（ジャンルメタ）→ pop/rock/jazz/folk/classical
#     - nottingham/ABC_cleaned/*.abc : 民謡（ABC, 1ファイル多曲）→ folk
#     - JSB chorales（music21内蔵）  : Bach四声 → classical（chordifyでコード、ソプラノを旋律）
#
#   出力:
#     - data/genres.corpus         : 8小節チャンク（genre, key, chords, melody）＝メロディモデル用
#     - data/genres_chords.corpus  : フル曲コード進行（genre, key, chords）＝コードモデルの構造学習用
#
#   使い方:  python3 tools/prepare_genres.py   （要 music21）

import os, sys, glob, sqlite3
import chordlib as cl

try:
    from music21 import converter, harmony, note, corpus, chord as m21chord
except ImportError:
    print("music21 が必要です:  pip install music21", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT_MEL = os.path.join(HERE, "data", "genres.corpus")
OUT_CH = os.path.join(HERE, "data", "genres_chords.corpus")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEYNAME = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
DURS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
CHUNK, MIN_BARS, MAX_BARS = 8, 4, 48

# music21 の chordKind → chordlib が解釈するクオリティ文字列
KIND2Q = {
    "major": "", "minor": "m", "augmented": "aug", "diminished": "dim",
    "dominant-seventh": "7", "major-seventh": "maj7", "minor-seventh": "m7",
    "half-diminished-seventh": "hdim", "diminished-seventh": "dim7",
    "minor-major-seventh": "mmaj7", "major-sixth": "6", "minor-sixth": "m6",
    "dominant-ninth": "9", "major-ninth": "maj9", "minor-ninth": "m9",
    "suspended-fourth": "sus4", "suspended-second": "sus2",
    "dominant-11th": "11", "dominant-13th": "13", "power": "",
}

# OpenEWLD のジャンル → アプリのジャンルラベル（対象外は None＝除外）
EWLD_MAP = {
    "Pop": "pop", "Rock": "rock", "Jazz": "jazz",
    "Folk, World, & Country": "folk", "Classical": "classical",
}


def midi_to_name(m):
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def quant_dur(x, remaining):
    x = min(x, remaining)
    cand = [d for d in DURS if d <= remaining + 1e-6] or [0.25]
    return min(cand, key=lambda d: abs(d - x))


def cs_to_roman(cs, tonic):
    root = cs.root()
    if root is None:
        return None
    qual = getattr(cs, "_q", None)              # _FakeCS（chordify）は _q を持つ
    if qual is None:
        qual = KIND2Q.get(cs.chordKind, "")
    bass_pc = None
    try:
        b = cs.bass()
        if b is not None and b.pitchClass != root.pitchClass:
            bass_pc = b.pitchClass              # スラッシュ（分数）コードのベース
    except Exception:
        pass
    return cl.chord_to_roman(root.pitchClass, tonic, qual, bass_pc)


def tonic_of(stream):
    try:
        k = stream.analyze("key")
    except Exception:
        return None
    pc = k.tonic.pitchClass
    if k.mode == "minor":
        pc = (pc + 3) % 12                       # 平行長調へ
    return pc


def measures_from_stream(mel_stream, chord_syms, tonic):
    """4/4前提。mel_stream=旋律(flat), chord_syms=[(offset,ChordSymbol)] → [(roman, barStr)]。"""
    tonic_ref = 60 + tonic
    notes = [n for n in mel_stream.getElementsByClass(note.Note)]
    if not notes:
        return []
    total_q = max(n.offset + n.quarterLength for n in notes)
    nbars = int(total_q // 4) + 1
    css = sorted(chord_syms, key=lambda x: x[0])
    out = []
    for m in range(nbars):
        m0, m1 = m * 4.0, (m + 1) * 4.0
        # 拍ごとに有効な最新コード（無い拍は直前を保持）
        beat_romans, prev_r = [], None
        for k in range(4):
            active = None
            for off, cs in css:
                if off <= m0 + k + 1e-6:
                    active = cs
                else:
                    break
            r = cs_to_roman(active, tonic) if active is not None else None
            beat_romans.append(r)
            if r is not None:
                prev_r = r
            elif prev_r is not None:
                beat_romans[-1] = prev_r
        firstnn = next((r for r in beat_romans if r is not None), None)
        if firstnn is None:
            continue
        beat_romans = [r if r is not None else firstnn for r in beat_romans]
        bar_notes = [n for n in notes if m0 - 1e-6 <= n.offset < m1 - 1e-6]
        if not bar_notes:
            continue
        bar_notes.sort(key=lambda n: n.offset)
        seq, cursor = [], 0.0
        for n in bar_notes:
            remaining = 4.0 - cursor
            if remaining <= 1e-6:
                break
            pos = n.offset - m0
            gap = pos - cursor
            if gap >= 0.25:
                rd = quant_dur(gap, remaining); seq.append(("R", rd)); cursor += rd
                remaining = 4.0 - cursor
                if remaining <= 1e-6:
                    break
            pitch = n.pitch.midi
            while pitch - tonic_ref > 24:
                pitch -= 12
            while pitch - tonic_ref < -12:
                pitch += 12
            dur = quant_dur(float(n.quarterLength), remaining)
            seq.append((midi_to_name(pitch), dur)); cursor += dur
        if cursor < 4.0 - 1e-6:
            seq.append(("R", round(4.0 - cursor, 3)))
        out.append((beat_romans, " ".join(f"{p}:{('%g' % d)}" for p, d in seq)))
        if len(out) >= MAX_BARS:
            break
    return out


def is_44(stream):
    ts = stream.getTimeSignatures()
    return bool(ts) and ts[0].ratioString == "4/4"


def extract_leadsheet(score):
    """リードシート(旋律＋ChordSymbol)の music21 スコア → [(roman,barStr)]。4/4のみ。"""
    flat = score.flatten()
    if not is_44(flat):
        return []
    tonic = tonic_of(flat)
    if tonic is None:
        return []
    css = [(cs.offset, cs) for cs in flat.getElementsByClass(harmony.ChordSymbol)]
    if not css:
        return []
    return measures_from_stream(flat, css, tonic), tonic


def extract_chorale(b):
    """JSBコラール → chordifyでコード、ソプラノを旋律に。4/4のみ。"""
    flat = b.flatten()
    if not is_44(flat):
        return []
    tonic = tonic_of(flat)
    if tonic is None:
        return []
    ch = b.chordify()
    css = []
    for c in ch.flatten().getElementsByClass(m21chord.Chord):
        r = c.root()
        if r is None:
            continue
        q = "m" if c.isMinorTriad() else ("dim" if c.isDiminishedTriad() else
             ("7" if c.isDominantSeventh() else ("maj7" if c.isMajorTriad() and len(c.pitches) > 3 else "")))
        css.append((c.offset, _FakeCS(r.pitchClass, q)))
    sop = b.parts[0].flatten()
    return measures_from_stream(sop, css, tonic), tonic


class _FakeCS:
    """chordify結果を cs_to_roman 互換にする簡易ラッパ。"""
    def __init__(self, root_pc, qual):
        self._pc = root_pc; self.chordKind = None; self._q = qual
    def root(self):
        class P: pitchClass = self._pc
        return P()


def cs_to_roman2(pc, qual, tonic):
    return cl.chord_to_roman(pc, tonic, qual)


def emit(song_bars, genre, title, out_mel, out_ch, stats):
    """[(beat_romans[4], barStr)] を melody 8小節チャンク＋フル chords に書き出す（保持トークン形式）。"""
    if len(song_bars) < MIN_BARS:
        return
    flat = " ".join(cl.to_holds([r for br, _ in song_bars for r in br]))
    out_ch.write(f"# {genre}: {title}\ngenre: {genre}\ntime: 4/4\nkey: C\nchords: " + flat + "\n\n")
    stats[genre] = stats.get(genre, 0)
    for c in range(0, len(song_bars), CHUNK):
        seg = song_bars[c:c + CHUNK]
        if len(seg) < MIN_BARS:
            continue
        out_mel.write(f"# {genre}: {title}#{c}\ngenre: {genre}\ntime: 4/4\nkey: C\n")
        out_mel.write("chords: " + " ".join(cl.to_holds([r for br, _ in seg for r in br])) + "\n")
        out_mel.write("melody: " + " | ".join(b for _, b in seg) + "\n\n")
        stats[genre] += 1


def run_ewld(out_mel, out_ch, stats):
    db = os.path.join(RAW, "OpenEWLD", "OpenEWLD.db")
    base = os.path.join(RAW, "OpenEWLD")
    if not os.path.exists(db):
        print("  (OpenEWLD なし・スキップ)"); return
    con = sqlite3.connect(db)
    rows = con.execute("SELECT w.title, w.path_leadsheet, g.genre FROM works w JOIN work_genres g ON w.id=g.id").fetchall()
    seen = set()
    for title, path, egenre in rows:
        genre = EWLD_MAP.get(egenre)
        if genre is None or path in seen:
            continue
        seen.add(path)
        fp = os.path.join(base, path)
        if not os.path.exists(fp):
            continue
        try:
            s = converter.parse(fp)
            res = extract_leadsheet(s)
            if res:
                emit(res[0], genre, title, out_mel, out_ch, stats)
        except Exception as ex:
            print(f"    ! EWLD skip {title}: {ex}", file=sys.stderr)
    print(f"  EWLD 完了 {stats}")


def run_nottingham(out_mel, out_ch, stats):
    files = sorted(glob.glob(os.path.join(RAW, "nottingham", "ABC_cleaned", "*.abc")))
    n = 0
    for f in files:
        try:
            op = converter.parse(f)
        except Exception:
            continue
        scores = op.scores if hasattr(op, "scores") else [op]
        for i, sc in enumerate(scores):
            try:
                res = extract_leadsheet(sc)
                if res:
                    emit(res[0], "folk", f"{os.path.basename(f)}#{i}", out_mel, out_ch, stats); n += 1
            except Exception:
                continue
    print(f"  Nottingham 完了: {n} 曲")


def run_jsb(out_mel, out_ch, stats):
    try:
        chorales = list(corpus.chorales.Iterator())
    except Exception as ex:
        print(f"  (JSB取得失敗: {ex})"); return
    n = 0
    for b in chorales:
        try:
            res = extract_chorale(b)
            if res:
                emit(res[0], "classical", b.metadata.title if b.metadata else "chorale", out_mel, out_ch, stats); n += 1
        except Exception:
            continue
    print(f"  JSB 完了: {n} 曲")


def main():
    os.makedirs(os.path.dirname(OUT_MEL), exist_ok=True)
    stats = {}
    with open(OUT_MEL, "w", encoding="utf-8") as out_mel, open(OUT_CH, "w", encoding="utf-8") as out_ch:
        out_mel.write("# 自動生成（prepare_genres.py）: 多ジャンルのメロディ＋コード（8小節チャンク）。\n\n")
        out_ch.write("# 自動生成（prepare_genres.py）: 多ジャンルのフル曲コード進行。\n\n")
        print("EWLD..."); run_ewld(out_mel, out_ch, stats)
        print("Nottingham..."); run_nottingham(out_mel, out_ch, stats)
        print("JSB..."); run_jsb(out_mel, out_ch, stats)
    print(f"\nメロディチャンク（ジャンル別）: {stats}")
    print(f"→ {os.path.relpath(OUT_MEL, HERE)} / {os.path.relpath(OUT_CH, HERE)}")


if __name__ == "__main__":
    main()
