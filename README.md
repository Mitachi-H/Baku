# Baku デモ用リポジトリ

![アプリの概要・要素技術](./images/図1.png)

上図の構成で開発を進めています。本リポジトリでは、提出用のモックコード一式を共有しています。

## セットアップ
```
pip install -r requirements.txt
```

## モック機能一覧

### 脳波データの検知モック `rem_analysis.py`
- REM 睡眠区間の検知と可視化を行います。
- 出力は `analysis_outputs/` 配下に保存されます。

使用例:
```
python rem_analysis.py --subject 0 --recording 1 --output-dir analysis_outputs
```

> 補足: PhysioNet などで公開されている睡眠中の脳波・EOG データを用いており、現在も REM 睡眠判定アルゴリズムを改良中です。

### 心拍データ解析モック `hr_analysis.py`
- 睡眠中の心拍データから心拍数や RR 間隔を推定します。
- 出力は `analysis_outputs/` に保存されます。

使用例:
```
python hr_analysis.py --subject 1 --output-dir analysis_outputs
```

### 眼球運動検知モック
- 予備実験で使用した Emotive 社の脳波計・EOG コードを公開しています。
- リポジトリ: https://github.com/Mitachi-H/dreamdive_emotive

### 心理療法チャットボット関連
- `demo_mini_llm_cbt.py`: CBT 向けチャットボットのファインチューニング用モック。
- `demo_llm.py`: 実行モック。以下のコマンドで Web UI がローカルに起動します。
```
python therapy_chat_mock.py
```
- GPT-4o API を利用するため、`.env` に `OPENAI_API_KEY=sk-...` を設定してください。
- DiaCBT、Cactus、DreamBank などの公開データセットを利用可能で、DDNSI 等の重症度指標の実装を進めています。

### 夢の動画化モック `dream_video_mock.py`
- 夢内容とポジティブな記憶を入力すると、音声・動画を生成し連結して再生します。
```
python dream_video_mock.py
```
- GPT-4o API を利用するため、`.env` に `OPENAI_API_KEY=sk-...` を設定してください。

### モックアプリケーション `demo_server.py`
- REM 睡眠を検知し、感覚刺激を再現するデモサーバーです。
```
python demo_server.py
```

## 補足
- ハードウェア構成については、別途提出資料を参照してください。
- 現在 LoRA を適用中のモジュールは動作イメージを示すコードのみであり、実際の応答品質は調整中です。
