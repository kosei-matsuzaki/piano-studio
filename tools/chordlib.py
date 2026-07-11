# chordlib.py — コード表記の解析・キー推定・ローマ数字化の共通処理。
# prepare_pop909.py / prepare_chords.py から使う。
# ここで出力するローマ数字トークンは neural.js の romanToChord() が解釈できる書式に限定する。

import re

BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# 半音(主音からの距離) → (ローマ数字, 臨時記号)。非ダイアトニックは近い度数＋♭/♯で表す。
CHROM_ROMAN = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", "b"), 4: ("III", ""),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", "b"), 9: ("VI", ""),
    10: ("VII", "b"), 11: ("VII", ""),
}


def to_holds(beat_tokens):
    """拍ごとのコード列 → 保持トークン形式（同じコードの継続を "_" に）。
       全コーパスのディスク上の書式を統一するため、prepare_* はこれを通して書き出す。"""
    out, prev = [], None
    for t in beat_tokens:
        if t == "_":
            out.append("_")
            continue
        out.append("_" if t == prev else t)
        prev = t
    return out


def note_pc(s):
    """'C', 'Gb', 'F#' → ピッチクラス(0-11)。不正なら None。"""
    m = re.match(r"^([A-Ga-g])([#b]*)$", s.strip())
    if not m:
        return None
    pc = BASE_PC[m.group(1).upper()]
    for c in m.group(2):
        pc += 1 if c == "#" else -1
    return pc % 12


DEG_SEMI = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}   # Harte/POP909 の /度数 → 根音からの半音


def _harte_bass_semi(s):
    """Harte のベース度数 '3','b7','#5','b3' → 根音からの半音。ダメなら None。"""
    m = re.match(r"^([#b]*)(\d+)$", s)
    if not m or int(m.group(2)) not in DEG_SEMI:
        return None
    v = DEG_SEMI[int(m.group(2))]
    for c in m.group(1):
        v += 1 if c == "#" else -1
    return v % 12


def quality_roman(q):
    """コードのクオリティ文字列 → (小文字にするか, ローマ数字の接尾辞)。
       9/11/13/6/add9/sus/aug/dim/半減 などのテンションを取りこぼさず解釈する。
       返す接尾辞は neural.js が解釈できる書式のみ。"""
    q = (q or "").strip()
    low = q.lower()
    core = re.sub(r"\([^)]*\)", "", q)          # 括弧を除いた本体（例 'm7(9)'→'m7'）
    corel = core.lower()
    # 付加テンション（Harte の *省略 は無視）。9/11/13/6 の有無を拾う。
    body = re.sub(r"\*[#b]?\d+", "", q)
    ext13 = bool(re.search(r"13", body))
    ext11 = bool(re.search(r"11", body))
    ext9 = bool(re.search(r"(?<!1)9", body)) or "add9" in low
    ext6 = bool(re.search(r"(?<!1)6", body))

    # --- 特殊クオリティ ---
    if "hdim" in low or "m7b5" in low or "m7-5" in low or ("m7" in corel and ("b5" in low or "-5" in low)) or "ø" in q:
        return (True, "ø7")
    if "dim" in corel:
        return (True, "°")
    if "aug" in corel or core.endswith("+"):
        return (False, "aug")
    if "sus2" in corel:
        return (False, "sus2")
    if "sus" in corel:
        return (False, "sus4")
    if "minmaj" in low or "mmaj" in low:
        return (True, "mM7")

    # --- メジャー/マイナー・7th の判定 ---
    maj7 = ("maj7" in low) or ("maj9" in low) or ("M7" in q) or ("M9" in q) or ("Δ" in q)
    is_min = corel.startswith("min") or corel.startswith("-") or (core[:1] == "m" and not corel.startswith("maj"))
    # 7th の有無：明示の '7'、maj7、または省略記法（C9/Cm9/CM9 など核が 9/11/13）＝7thを含む。
    core_num = re.sub(r"^(maj|min|dim|aug|sus2|sus4|sus|hdim|half|dom|m|M|\+|-)+", "", corel).strip()
    shorthand7 = bool(re.match(r"^(9|11|13)", core_num)) and "add" not in low
    has7 = maj7 or (re.search(r"(?<![#b\d])7", core) is not None) or shorthand7
    dom7 = (not maj7) and has7

    if is_min:
        if maj7:
            return (True, "mM7")
        if has7 and (ext13 or ext11):
            return (True, "11")
        if has7 and ext9:
            return (True, "9")                  # m7(9)=min9
        if ext9 and not has7:
            return (True, "add9")               # m(9)=minor add9
        if has7:
            return (True, "7")                  # min7
        if ext6:
            return (True, "6")                  # m6
        return (True, "")
    if maj7:
        return (False, "maj9") if (ext9 or ext11 or ext13) else (False, "maj7")
    if dom7:
        if ext13:
            return (False, "13")
        if ext11:
            return (False, "11")
        if ext9:
            return (False, "9")
        return (False, "7")
    if ext6:
        return (False, "6")
    if ext9:
        return (False, "add9")                  # 7th無しの9 ＝ add9
    return (False, "")


# 根音以外のベースはすべて '/半音(主音相対0-11)' で保持する。転回形（G/B, C/E等の
# コードトーンベース）はベースライン進行としてJ-POP/ポップスで重要なため落とさない。
# 生データではスラッシュコードが12〜19%を占める。語彙の肥大はレア分数トークンを
# 学習時に素のコードへ落とす頻度フィルタ（train.py load_all_corpora）で抑える。
_INVERSION_INTERVALS = {0}                      # 根音ベースのみ冗長として省略


def _bass_suffix(root_pc, bass_pc, tonic_pc):
    """ベース音を '/半音(0-11)' にする（根音と同じベースだけ省略）。"""
    if bass_pc is None:
        return ""
    if (bass_pc - root_pc) % 12 in _INVERSION_INTERVALS:
        return ""
    return "/" + str((bass_pc - tonic_pc) % 12)


def chord_to_roman(root_pc, tonic_pc, quality, bass_pc=None):
    """根音pc・主音pc・クオリティ(・ベースpc) → ローマ数字トークン
       （例 'V7', 'vi', 'bVII', 'iiø7', 'IVmaj9', 'V7/11'）。"""
    semi = (root_pc - tonic_pc) % 12
    numeral, acc = CHROM_ROMAN[semi]
    lower, suffix = quality_roman(quality)
    if lower:
        numeral = numeral.lower()
    return acc + numeral + suffix + _bass_suffix(root_pc, bass_pc, tonic_pc)


def parse_chord_label(label):
    """コード表記 → (root_pc, quality, bass_pc)。bass_pc は分数コードのベース（無ければ None）。
       対応: Harte 'C:maj/3'・'G:sus4(b7)'、連結 'Am7(9)'・'A/C#'、度数/音名どちらのベースも。"""
    label = label.strip()
    if not label or label.upper() in ("N", "X", "NC", "N.C."):
        return None
    bass_raw = None
    if "/" in label:                                   # ベース（分数コード）を分離
        label, bass_raw = label.rsplit("/", 1)
    if ":" in label:                                   # 'Root:quality'（Harte / POP909）
        root, qual = label.split(":", 1)
    else:                                              # 'Am7' 連結形式
        m = re.match(r"^([A-Ga-g][#b]*)(.*)$", label)
        if not m:
            return None
        root, qual = m.group(1), m.group(2)
    root_pc = note_pc(root)
    if root_pc is None:
        return None
    bass_pc = None
    if bass_raw:
        deg = _harte_bass_semi(bass_raw)               # '/3' '/b7' … 度数ベース
        bass_pc = (root_pc + deg) % 12 if deg is not None else note_pc(bass_raw)  # '/C#' 音名ベース
    return (root_pc, qual, bass_pc)


def key_tonic_pc(label):
    """'Gb:maj', 'A:min', 'C' → 主音pc（短調は平行長調の主音に変換）。不明は None。"""
    label = (label or "").strip()
    if not label or label.upper() in ("N", "X"):
        return None
    parts = label.split(":")
    pc = note_pc(parts[0])
    if pc is None:
        return None
    mode = parts[1].lower() if len(parts) > 1 else "maj"
    if mode.startswith("min") or mode in ("aeolian", "a"):
        pc = (pc + 3) % 12                  # 平行長調
    return pc


def estimate_tonic(chord_events):
    """[(root_pc, quality, weight)] からダイアトニック被覆が最大の長調主音を推定。"""
    best_t, best_score = 0, -1.0
    for t in range(12):
        diat = {(t + iv) % 12 for iv in MAJOR_SCALE}
        score = 0.0
        for root_pc, qual, w in chord_events:
            if root_pc in diat:
                score += w
            semi = (root_pc - t) % 12
            if semi == 0:                    # 主和音は加点
                score += 0.5 * w
            if semi == 7:                    # 属和音も加点
                score += 0.3 * w
        if score > best_score:
            best_score, best_t = score, t
    return best_t
