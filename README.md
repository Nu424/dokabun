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
   OpenAI 互換エンドポイントに切り替える例:
   ```bash
   python -m dokabun -i examples/sample.csv --model gpt-4o-mini --base-url https://api.openai.com/v1
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
   | `--base-url` | LLM API のベース URL (既定: `https://openrouter.ai/api/v1`) |
   | `--concurrency` | 同時並列数 (既定: 5) |
   | `--partial-interval` | 何行ごとに一時保存するか |
   | `--max-text-file-bytes` | テキストファイル読み込み時の最大サイズ（バイト）(既定: 262144) |
   | `--timestamp` | 途中再開したいタイムスタンプを明示指定 |
   | `--nsf-ext` | `nsf_` 列の出力拡張子 (`txt` / `md`) |
   | `--nsf-name-template` | `nsf_` 通常時のファイル名テンプレ |
   | `--nsf-name-template-filetarget` | `t_` が1列かつファイルパス時のファイル名テンプレ |

3. 実行すると、`output_dir`（未指定時は入力ファイルと同じディレクトリ）に下記ファイルが出力されます。

   - `input_YYYYMMDD_HHMMSS.xlsx` … 元ファイルのコピー
   - `input_YYYYMMDD_HHMMSS.out.xlsx` … 処理後の最終出力
   - `input_YYYYMMDD_HHMMSS.partial.*.xlsx` … 一時保存ファイル
   - `input_YYYYMMDD_HHMMSS.meta.json` … 再開用メタデータ

   メタファイルが存在する場合、CLI は最新タイムスタンプを自動で選択し途中再開します。  
   新規実行したい場合は `--timestamp` を指定するか、既存メタファイルを削除してください。

## 開発者向け: 実行フロー（関数呼び出し）

### 入口〜全体処理（CLI → runner）

- `python -m dokabun` (`dokabun/__main__.py`)
  - _CLIの処理_
  - `dokabun.cli.main(argv)`
    - `build_parser()` / `argparse` で引数を解釈
    - `load_dotenv()`（`.env` があれば読み込み）
    - `_build_config_from_args(args)` → `AppConfig.from_dict(...)`
      - `--timestamp` 未指定時は `_detect_latest_timestamp(...)` で `output_dir` 内の `*.meta.json` から最新を推定
    - `configure_logging(...)` / `_load_api_key()`
    - _runner_
    - `dokabun.core.runner.run(config, api_key)`
      - `asyncio.run(run_async(config, api_key))`
        - _再開処理_
        - `SpreadsheetReaderWriter(config)` で `timestamp` を確定し、入出力パスを準備
        - `reader.load_meta_if_exists()` → `last_completed_row` を読み、再開位置 `start_row_index` を決定
        - _スプシの読み込み・列の分割_
        - `df = reader.load()`
          - `{stem}_{timestamp}.partial.*.xlsx` があれば最新を優先して読み込み
          - 無ければ入力を `{stem}_{timestamp}{ext}` にコピーして読み込み（`.xlsx` / `.csv`）
        - `_classify_columns(df)` で列を分類（`t_` / 構造化出力 / `ns_`・`nsf_`）
        - `_collect_work_items(...)` で「未入力セルが残る行」だけを `RowWorkItem` として列挙
        - `AsyncOpenRouterClient(...)` と `asyncio.Semaphore(max_concurrency)` を用意
        - 各行をタスク化し、`asyncio.as_completed(...)` で完了順に回収
          - 1行毎の処理は、`process_row_async(...)` で行う
          - 結果は `df` に反映しつつ `ExecutionSummary` に成功/失敗・usage を集計
          - `partial_interval` 行ごとに `reader.save_partial(df, start_row, end_row)`（一時保存＋meta更新）
        - 最後に `reader.save_output(df)` で `{stem}_{timestamp}.out.xlsx` を保存し、サマリーを標準出力へ表示

### 1 行の処理（`process_row_async`）

- **ターゲット列の前処理**: `_build_targets(...)` → `run_preprocess_pipeline(...)`
  - `ImagePreprocess` / `TextFilePreprocess` / `PlainTextPreprocess` の順で判定し、`Target`（`TextTarget`/`ImageTarget`）へ変換
  - ファイルパスは `SpreadsheetReaderWriter.target_base_dir`（= 入力スプレッドシートのディレクトリ）からの相対として解決
- **構造化出力（通常の出力列）**
  - `build_schema_from_headers(pending_columns, name=...)` で LLM に渡す JSON Schema を生成
    - `列名|説明` の場合も **`列名` 部分**をプロパティ名として列へマッピング
  - `build_prompt(...)` → `client.create_completion(...)`（OpenRouter 経由で構造化出力を要求）
  - 応答は `_extract_parsed_json(...)` で JSON 化し、対応する出力列に値を書き戻す
- **非構造化出力（`ns_` / `nsf_`）**
  - `_parse_ns_prompt(column)` で列名からプロンプト文を抽出 → `build_nonstructured_prompt(...)` → `client.create_completion_text(...)`
  - `nsf_` は `output_dir` にファイル保存し、セルには相対ファイル名を書き込み（命名は `--nsf-name-template*` で制御）

### 途中再開の考え方（`timestamp`）

- 再開時は、`{stem}_{timestamp}.meta.json` の `last_completed_row` と、`{stem}_{timestamp}.partial.*.xlsx`（存在すればそれも）を基点に続きから処理します。
- ただし各セルは `_is_empty_value(...)` で「空」と判定されたときだけ埋めるため、同じ `timestamp` で再実行しても既存値は基本的に上書きしません。

## スプレッドシート形式

- 列名が `t_` で始まる列を **分析対象列** として扱います。複数ある場合は左から順にプロンプトへ渡されます。
  - セルの値がテキストファイルパス（`.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yml`, `.yaml`）の場合、そのファイルを読み込んで内容を LLM に渡します。
  - セルの値が画像ファイルパス（`.png`, `.jpg`, `.jpeg`, `.webp`）の場合、その画像を LLM に渡します。
  - それ以外はプレーンテキストとして扱います。
  - ファイルパスはスプレッドシートファイルからの相対パスとして解釈されます。
- それ以外の列は **出力列** で、ヘッダを `列名` または `列名|説明` と記述します。
  - `説明` は JSON Schema の `description` に利用されます。
- LLM は空欄の出力列のみを埋めます。既に値が入っているセルは上書きしません。

### 非構造化出力列 (`ns_`, `nsf_`)

- `ns_{プロンプト}`: 列ごとに非構造化（プレーンテキスト）出力を生成し、セルへ書き込みます。まとめずに列単位で呼び出します。
- `nsf_{プロンプト}`: 生成結果をファイルへ保存し、セルにはファイル名（`output_dir` 相対）を書き込みます。
  - デフォルト名: `nsf{index}_{行番号}.txt`（行番号はデータ行 1 始まり）
    - 例: `nsf1_1.txt`, `nsf1_2.txt`, `nsf1_3.txt`, ...
  - `t_` 列が 1 つで、セルが実在ファイルパスなら `{target_file_stem}_nsf{index}.txt` を採用
    - 例: `text01.txt` -> `text01_nsf1.txt`

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
