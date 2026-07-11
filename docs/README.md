# docs — 資料

| ファイル | 内容 |
|----------|------|
| [`PORTFOLIO.md`](PORTFOLIO.md) | **作品紹介資料**（概要・スクリーンショット・技術構成・AI利用・権利）。 |
| [`curriculum.md`](curriculum.md) | **全52パート・4章立て（準備／メロディ／コード／リズム編）の正本（解説テキスト＋スライド構成＋出典）**。内容を直すときはここで方針を決める。 |
| [`data-pipeline.md`](data-pipeline.md) | AI作曲モデルのデータ取得〜前処理〜学習の全手順。 |

## 内容の直し方（ブラッシュアップ手順）

1. `docs/curriculum.md` で該当パート／スライドの記述を直す。
2. アプリの実データ `../src/curriculum.js`（同じ構成）に反映する。
3. `node --check src/curriculum.js` で構文確認 → ブラウザで `../index.html` を再読み込み。

## スライドの書き方（`curriculum.js` のスキーマ）

```js
{ title: "Part N タイトル", slides: [
  {
    heading: "見出し",
    body: ["箇条書き1", "箇条書き2"],
    staff: ["C4","E4","G4"],            // 五線譜の音符。各要素は↓も可
    // 和音（縦に積む）: ["C4","E4","G4"] / {chord:["C4","E4","G4"], label:"C"}
    // コードから自動で正しい綴り: {chord:{root:"F",type:"min"}, label:"Fm"} → F A♭ C
    staffOpts: { solfege:true, brackets:[{from:0,to:1,text:"3度"}] },
    key: "G",                            // 下部ピアノの左端の音（調）を自動設定
    keys: ["G4","B4","D4"],              // 下部ピアノでハイライト（ピッチクラス一致）
    voicing: [4,7,12],                   // 実音の積み方をハイライト（C基準オフセット）
    // ※ 練習でないスライドは、staff / keys / voicing の内容を
    //   楽譜＋下部ピアノで自動アニメーション（音符・鍵盤が順に光る＋音）で説明する。
    //   （旧 listen ボタン・鍵盤ヒント文言は廃止。listen フィールドは未使用）
    practice: { chords:[{root:"C",type:"maj"}], label:"…" },
    // practice は notes:[offset] / chords:[{root,type}] / voicings:[[offset...]] のいずれか
  },
]}
```

- **鍵盤の解説・練習は画面下部のピアノを使う**（別の鍵盤画像は作らない）。
- スライド／練習に `key` があると、下部ピアノの調が**自動で切り替わる**。
- 使えるコードタイプ（`type`）：`maj min dim aug sus4 sus2 maj7 min7 dom7 m7b5 six m6 mM7 add9`。
  これ以外の響き（9th/11th/13th/オルタード等）は `keys`＋`listen` か `voicings`（オフセット）で表現する。
- 練習の `voicings`／`notes` のオフセットは **0〜19 がPCキーで弾ける範囲**（それ以上はマウス／ハイライトのみ）。
- **リズム編**は `rhythm:{tempo,beats,bars,grid,loop,layers:[{name,pitch,accentPitch,hits,accents,drum}]}`。拍は**ドラム音源**（選択楽器と独立）で鳴らし、`pitch`（または `drum`）でキック/スネア/ハイハット/クラーベ/クリックを選ぶ。詳細は `curriculum.md`。
- **応用例メロディ**は `example:{tempo,chords,melody,label}`（主音相対オフセット）を**付けたスライドだけ**再生（スケール・旋法・進行など「使うとどう聞こえるか」が要点の回に用意。全スライドには付けない）。
- **概念デモ（鍵盤再生）**は自動判定：音符・和音が2つ以上／4音以上のリッチ和音のときだけ鳴らし、**単音・単一トライアド・音材なしは楽譜のみ**。`anim:true/false` で上書き可。
- **章立て**は各章の**先頭パート**に `chapter:"準備編"` 等を付ける（`renderList()` が見出しを挿入）。

## 出典

正確性確認のため参照したページは `curriculum.md` 末尾の「出典」に記載。
（リレイテッドⅡm7、裏コード、パッシングディミニッシュ、サブドミナントマイナー、分数aug、10th/12th/14th テンション 等）
