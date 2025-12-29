# dokabun

スプレッドシートをインターフェースとして LLM の構造化出力を活用する CLI ツールです。  
`t_` で始まる列を分析対象として読み込み、空欄になっている出力列を OpenRouter 経由で自動的に埋めます。

## 使い方

1. OpenRouter の API キーを取得し、環境変数に設定します。

   ```powershell
   setx OPENROUTER_API_KEY "sk-..."
   ```

2. 依存パッケージをインストールし、CLI を実行します。

   ```bash
   uv sync  # もしくは pip install -e .
   python -m dokabun -i examples/sample.csv --model openai/gpt-4.1-mini
   ```
   or (uv を使っている場合)
   ```bash
   uv run python -m dokabun -i examples/sample.csv --model openai/gpt-4.1-mini
   ```

   主なオプション:

   | オプション | 説明 |
   | --- | --- |
   | `-i, --input` | 入力スプレッドシート (`.xlsx` または `.csv`) |
   | `--model` | 使用するモデル名 (既定: `openai/gpt-4.1-mini`) |
   | `--concurrency` | 同時並列数 (既定: 5) |
   | `--partial-interval` | 何行ごとに一時保存するか |
   | `--max-text-file-bytes` | テキストファイル読み込み時の最大サイズ（バイト）(既定: 262144) |
   | `--timestamp` | 途中再開したいタイムスタンプを明示指定 |

3. 実行すると、`output_dir`（未指定時は入力ファイルと同じディレクトリ）に下記ファイルが出力されます。

   - `input_YYYYMMDD_HHMMSS.xlsx` … 元ファイルのコピー
   - `input_YYYYMMDD_HHMMSS.out.xlsx` … 処理後の最終出力
   - `input_YYYYMMDD_HHMMSS.partial.*.xlsx` … 一時保存ファイル
   - `input_YYYYMMDD_HHMMSS.meta.json` … 再開用メタデータ

   メタファイルが存在する場合、CLI は最新タイムスタンプを自動で選択し途中再開します。  
   新規実行したい場合は `--timestamp` を指定するか、既存メタファイルを削除してください。

## スプレッドシート形式

- 列名が `t_` で始まる列を **分析対象列** として扱います。複数ある場合は左から順にプロンプトへ渡されます。
  - セルの値がテキストファイルパス（`.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yml`, `.yaml`）の場合、そのファイルを読み込んで内容を LLM に渡します。
  - セルの値が画像ファイルパス（`.png`, `.jpg`, `.jpeg`, `.webp`）の場合、その画像を LLM に渡します。
  - それ以外はプレーンテキストとして扱います。
  - ファイルパスはスプレッドシートファイルからの相対パスとして解釈されます。
- それ以外の列は **出力列** で、ヘッダを `列名` または `列名|説明` と記述します。
  - `説明` は JSON Schema の `description` に利用されます。
- LLM は空欄の出力列のみを埋めます。既に値が入っているセルは上書きしません。

`examples/sample.csv` には最小構成のサンプルが含まれています。

| t_content | summary\|本文の要約 | sentiment\|感情ラベル |
| --- | --- | --- |
| 今日はとても暑い一日でした。 | (空欄) | (空欄) |
| 新商品のレビューを集めています。 | (空欄) | (空欄) |

## テスト

主要コンポーネントには `pytest` ベースのテストが付属しています。

```bash
pytest
```

Fake クライアントを用いてネットワークアクセスなしで `process_row_async` を検証しています。

## ライセンス

MIT License
