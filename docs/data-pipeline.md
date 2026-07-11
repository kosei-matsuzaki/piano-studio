# AI作曲 データパイプライン

「AI作曲」の学習済みモデル `model.js` を作るためのデータ取得〜前処理〜学習の全手順。
すべて `tools/` 配下で完結する。ブラウザ側(`neural.js` / `compose.js`)はこの成果物を読むだけ。

## 全体像

```
tools/fetch_data.sh                 公開データセットを data/raw/ へ取得
        │
        ▼  前処理（データセットごとに1本、共通コーパス形式へ変換）
prepare_pop909.py      → data/pop909.corpus        (メロディ+コード, 8小節チャンク×11,072)
                       → data/pop909_chords.corpus (フル曲コード進行×880, 構造学習用)
prepare_chords.py      → data/chords_only.corpus   (コードのみ×429, 拍数は秒から推定)
prepare_genres.py      → data/genres.corpus        (多ジャンル メロディ+コード×3,193)
                       → data/genres_chords.corpus (多ジャンル フル進行×1,208)
prepare_pop909_dyn.py  → data/pop909_dyn.npz       (小節特徴→実演奏ベロシティ, 69,183小節)
        │
        ▼  学習（model.js の該当ブロックを差し替え）
train_chord_tf.py      → model.js "chord"  : コード進行 Transformer（フル曲進行で学習）
train_mel_tf.py        → model.js "melody" : メロディ Transformer（コード/拍/フレーズ/ジャンル条件）
train_dyn.py           → model.js "dyn"    : 強弱曲線 LSTM（曲調→小節ごとの強さ）
```

## ゼロから作り直す手順

```bash
bash tools/fetch_data.sh                 # データ取得（要ネット。POP909が数百MB）
pip install pretty_midi music21 torch numpy

# 前処理（すべて data/*.corpus / *.npz を上書き生成）
python3 tools/prepare_pop909.py
python3 tools/prepare_chords.py
python3 tools/prepare_genres.py          # music21 使用。数十分かかる
python3 tools/prepare_pop909_dyn.py

# 学習（CPUで各1〜2時間）。model.js が無い初回だけ train_torch.py で雛形を作る
python3 tools/train_torch.py             # 初回のみ: model.js の骨格(LSTM版)を生成
python3 tools/train_chord_tf.py --epochs 90
python3 tools/train_mel_tf.py --epochs 40
python3 tools/train_dyn.py
```

既存の model.js があれば `train_torch.py` は不要（各 train_*_tf.py は該当ブロックだけ差し替える）。

## データセット一覧（data/raw/）

| データセット | 内容 | 前処理 | 貢献 |
|---|---|---|---|
| POP909-Dataset | 中国ポップス909曲のピアノカバーMIDI（MELODY/BRIDGE/PIANOトラック、拍・コード・キー注釈、実演奏ベロシティ） | prepare_pop909.py / prepare_pop909_dyn.py | メロディ・コード・強弱の主力 |
| Chord-Dataset (anime-song) | アニソン138曲のコード進行（秒タイムスタンプ付き） | prepare_chords.py | コード（彩り・分数が豊富） |
| Chord-Annotations (tmc323) | RWC Pop 100曲＋uspop 195曲のコード注釈(.lab) | prepare_chords.py | コード |
| OpenEWLD | リードシート集（MusicXML、ジャンルDB付き） | prepare_genres.py | 多ジャンルのメロディ+コード |
| nottingham | 英国民謡1,200曲（ABC、コード付き） | prepare_genres.py | folk（4/4のみ採用） |
| JSBコラール | music21 内蔵（取得不要） | prepare_genres.py | classical |
| McGill-Billboard | 米ポップス890曲のコード注釈(.lab、mirdata公式ミラー) | prepare_chords.py | コード（実際のヒット曲の進行） |
| isophonics | Beatles 180曲+Queen/Carole King/Zweieck のコード注釈(.lab) | prepare_chords.py | コード（ロック） |
| wikifonia-mxl | Wikifonia リードシート約6,400曲（.mxl。Wayback Machine 経由） | prepare_wikifonia.py | メロディ+コードの大規模増強（genre=songbook） |

## コーパスの住み分け（どのファイルがどのモデルに使われるか）

コーパスは**「melody: 行があるか」で用途が分かれる**。ファイル名の規約とあわせて:

| ファイル | melody行 | 使うモデル | 内容 |
|---|---|---|---|
| `pop909.corpus` / `genres.corpus` | **あり** | **メロディTF**（train_mel_tf.py） | 8小節チャンク。chords行は「その音が鳴っている拍のコード」という**メロディの文脈**としてだけ使われる |
| `pop909_chords.corpus` / `genres_chords.corpus` / `chords_only.corpus` | なし | **コードTF**（train_chord_tf.py） | **フル曲**のコード進行。曲の構造（反復・展開・終止）を学ばせるため、チャンクではなく全曲を1系列で持つ |
| `pop909_dyn.npz` | — | 強弱LSTM（train_dyn.py） | corpus形式ではなく小節特徴+実演奏ベロシティの数値データ |

つまり同じPOP909から「メロディ用（チャンク・melody付き）」と「コード用（フル曲・melodyなし）」の
**2つのファイルが別々に**出力される。トレーナ側の選別は `melody` の有無で行う
（train_chord_tf.py: `not s.get("melody")` / train_mel_tf.py: build_dataset の mel_seqs）。
コードモデルにチャンクを食わせると通作的になり、メロディモデルにフル曲は長すぎる（MAXPOS超過）ための分離。

## 共通コーパス形式（*.corpus）

全データ源はこの形式に変換される。`train.py parse_corpus` が唯一のリーダー。

```
# 曲名コメント
genre: pop          ← 任意（省略時 pop）
time: 4/4           ← 任意（省略時 4/4。現状 4/4 のみ）
key: C              ← メロディの調（コードはローマ数字なので調に依存しない）
chords: I _ _ _ IV _ V _    ← 1トークン=1拍。"_"=直前コードの保持（保持トークン形式）
melody: C5:1 D5:1 E5:1 G5:1 | ...   ← 小節を「|」区切り、音名:拍数。省略可（コード専用データ）
```

- 旧形式（1小節1コード、拍ごと繰り返し）も読み込み時に保持形式へ自動正規化される
  （2026-07-10 以降、全 prepare_* は保持形式で書き出すためディスク上も統一済み）
- 分数コードは `/半音(主音相対0-11)` サフィックス。レアな分数（出現<30回）は学習時にベースを自動省略
- メロディは学習時にトニック相対へ正規化されるため全キーで生成可能
- 完全重複エントリ（曲内の繰り返しセクション由来）は読み込み時に自動除去される
- 品質検査: `python3 tools/validate_corpus.py`（前処理を変えたら必ず実行。書式・拍数・語彙の汚れ・重複を検査）

## 各スクリプトの役割

| ファイル | 役割 |
|---|---|
| `chordlib.py` | コード表記解析・キー推定・ローマ数字化（前処理の共通部品） |
| `train.py` | コーパス読込・語彙・データ構築の**共通ライブラリ**（+numpy参照トレーナ） |
| `train_torch.py` | LSTM版トレーナ（初回の model.js 雛形生成用。export_lstm は train_dyn が利用） |
| `train_chord_tf.py` | コード進行 Transformer（dim160×3層。保持トークン・フル曲進行で構造を学習） |
| `train_mel_tf.py` | メロディ Transformer（dim128×3層。コード/小節内拍位置/フレーズ位置/ジャンル条件） |
| `train_dyn.py` | 強弱曲線 LSTM（H48。小節特徴24次元→曲内zベロシティ。※torch.set_num_threads(4)必須） |

## 学習後の確認

- 各トレーナは最後に「PyTorch vs numpy(export) 最大誤差」を表示する（ブラウザ推論との一致検証。0.01未満なら正常）
- 生成品質の定量指標（拍間変化率≈0.3、強拍整列、保持長≤8拍、彩り率≈10%、モチーフ反復率など）は
  生成時バイアス（neural.js）とセットでチューニングされている。モデルを差し替えたら
  アプリで数曲生成して、進行の偏り・保持長・終止を確認すること。

## データ拡張の履歴と候補

**導入済み（2026-07-11）**: McGill Billboard・isophonics（→ chords_only が429→1,527曲）、
Wikifonia（→ wikifonia.corpus / wikifonia_chords.corpus、genre=songbook）。
Wikifonia は OpenEWLD の上位集合のため重複が出るが、完全一致は読込時に自動除去される。
配布元が消失しているため fetch_data.sh は Wayback Machine のスナップショットから取得する。

**残りの候補**:
1. **iRb / ジャズコーパス** — ジャズスタンダード約1,300曲のコード進行。ジャンル多様性向け。
2. Hooktheory (TheoryTab) — セクションラベル付きで理想的だが全量はスクレイピング要のため保留。
3. RS200 / CoCoPops — メロディ+和声の書き起こし（humdrum形式。変換コスト中）。
