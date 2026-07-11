# AI作曲モデルの学習（tools/）

ブラウザの「AI作曲」で使う学習済みモデル `../src/model.js` を作るオフライン学習ツール一式。

**アーキテクチャ**: コード進行 Transformer → コード条件つきメロディ Transformer ＋ 強弱曲線 LSTM の3ブロック。
学習は PyTorch、重みは `model.js`（`window.COMPOSER_MODEL`）に JSON で書き出し、
ブラウザ側は依存なしの自作推論エンジン `../src/neural.js` が読んで生成する（拍位相・彩り・理論・周期性・
語彙予算などの生成時バイアスも neural.js 側にある）。

```
data/raw/（公開データセット） ─ prepare_*.py → data/*.corpus / *.npz
        └ train_chord_tf.py / train_mel_tf.py / train_dyn.py → ../src/model.js
```

**手順・データセット一覧・コーパス形式・拡張候補の詳細は `../docs/data-pipeline.md` を参照。**

クイックリファレンス:

```bash
bash tools/fetch_data.sh                                  # データ取得
python3 tools/prepare_pop909.py && python3 tools/prepare_chords.py \
  && python3 tools/prepare_genres.py && python3 tools/prepare_pop909_dyn.py   # 前処理
python3 tools/train_chord_tf.py --epochs 90               # コード進行（約1時間/CPU）
python3 tools/train_mel_tf.py --epochs 40                 # メロディ（約1.5時間/CPU）
python3 tools/train_dyn.py                                # 強弱曲線（数分）
```

- `corpus.txt` … 手書きのスターターコーパス（コーパス形式の説明つき）。曲を追記できる。
- `chordlib.py` … コード表記解析・キー推定・ローマ数字化（前処理の共通部品）。
- `train.py` … コーパス読込・語彙構築の共通ライブラリ（全トレーナが import）。
- `train_torch.py` … LSTM版トレーナ。model.js をゼロから作る場合の雛形生成にのみ使用。
