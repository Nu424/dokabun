# dokabun

スプレッドシートをインターフェースとして LLM の構造化/非構造化出力を活用する CLI ツールです。  
`i_` で始まる列を分析対象として読み込み、空欄になっている出力列を OpenRouter 経由で自動的に埋めます。

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
| `--nsof-ext` | `nsof_` 列の出力拡張子 (`txt` / `md`) |
| `--nsof-name-template` | `nsof_` 通常時のファイル名テンプレ |
| `--nsof-name-template-filetarget` | `i_` が1列かつファイルパス時のファイル名テンプレ |

3. 実行すると、`output_dir`（未指定時は入力ファイルと同じディレクトリ）に下記ファイルが出力されます。

   - `input_YYYYMMDD_HHMMSS.xlsx` … 元ファイルのコピー
   - `input_YYYYMMDD_HHMMSS.out.xlsx` … 処理後の最終出力
   - `input_YYYYMMDD_HHMMSS.partial.*.xlsx` … 一時保存ファイル
   - `input_YYYYMMDD_HHMMSS.meta.json` … 再開用メタデータ
   - `input_YYYYMMDD_HHMMSS.generations.jsonl` … LLM 呼び出しの generation_id ログ
   - `input_YYYYMMDD_HHMMSS.generation_costs.jsonl` … `/generation` 再取得キャッシュ
   - `eof{index}_{row}.npy` … 埋め込みベクトル（`eof` / フォールバック時）

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
        - `_classify_columns(df)` で列を分類（`i_` / `so_` / `nso_`・`nsof_` / `l_` / `eo*`）
          - `ColumnClassification` にまとめて返却
          - 大小文字どちらでも可
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
- **構造化出力（`so_` 列）**
  - `so_{name}|説明` から `name|説明` を組み立てて `build_schema_from_headers(...)` で JSON Schema を生成
    - `so_` を除去した **`name` 部分**をプロパティ名として列へマッピング
  - `build_prompt(...)` → `client.create_completion(...)`（OpenRouter 経由で構造化出力を要求）
  - 応答は `_extract_parsed_json(...)` で JSON 化し、対応する出力列に値を書き戻す
- **非構造化出力（`nso_` / `nsof_`）**
  - `_parse_ns_prompt(column)` で列名からプロンプト文を抽出 → `build_nonstructured_prompt(...)` → `client.create_completion_text(...)`
  - `nsof_` は `output_dir` にファイル保存し、セルには相対ファイル名を書き込み（命名は `--nsof-name-template*` で制御）

### 途中再開の考え方（`timestamp`）

- 再開時は、`{stem}_{timestamp}.meta.json` の `last_completed_row` と、`{stem}_{timestamp}.partial.*.xlsx`（存在すればそれも）を基点に続きから処理します。
- ただし各セルは `_is_empty_value(...)` で「空」と判定されたときだけ埋めるため、同じ `timestamp` で再実行しても既存値は基本的に上書きしません。
- LLM 呼び出しの **課金コストは `/generation` で後から確定**させるため、generation_id を JSONL で永続化し、実行末尾にまとめて再取得します。`--timestamp` を指定した再実行でもコストだけ再計算されます（行処理がなくてもコスト再計算だけ動きます）。

## スプレッドシート形式

- **入力列**: `i_` で始まる列を分析対象として扱います（大小文字不問）。複数ある場合は左から順にプロンプトへ渡されます。
  - セルの値がテキストファイルパス（`.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yml`, `.yaml`）の場合、そのファイルを読み込んで内容を LLM に渡します。
  - セルの値が画像ファイルパス（`.png`, `.jpg`, `.jpeg`, `.webp`）の場合、その画像を LLM に渡します。
  - それ以外はプレーンテキストとして扱います。
  - ファイルパスはスプレッドシートファイルからの相対パスとして解釈されます。
- **ラベル列**: `l_` で始まる列は処理に影響しません。後段の解析用メタデータとして利用できます。
- **構造化出力列**: `so_{列名}[|説明]`
  - JSON Schema のプロパティ名は `so_` を除去した `{列名}` を使用します。
- LLM は空欄の出力列のみを埋めます。既に値が入っているセルは上書きしません。

### 非構造化出力列 (`ns_`, `nsf_`)

- `nso_{プロンプト}`: 列ごとに非構造化（プレーンテキスト）出力を生成し、セルへ書き込みます。まとめずに列単位で呼び出します。
- `nsof_{プロンプト}`: 生成結果をファイルへ保存し、セルにはファイル名（`output_dir` 相対）を書き込みます。
  - デフォルト名: `nsf{index}_{行番号}.txt`（行番号はデータ行 1 始まり）
    - 例: `nsof1_1.txt`, `nsof1_2.txt`, `nsof1_3.txt`, ...
  - `t_` 列が 1 つで、セルが実在ファイルパスなら `{target_file_stem}_nsf{index}.txt` を採用
    - 例: `text01.txt` -> `text01_nsof1.txt`

### 埋め込み出力列 (`eo*`)

`eo` で始まる列は **埋め込み(Embedding)ベクトル**を作成して出力します。

- **前提**: `--model` は埋め込み対応モデルを指定してください（例: `openai/text-embedding-3-small`）。  
  ※ `--model` をチャット補完と埋め込みで共用しているため、同一実行で両方やる場合はモデル選択に注意してください。
- **埋め込み入力**: `i_` 列のうち `TextTarget` になったものを左→右で連結（区切りは `\n\n`）。画像は埋め込み入力から除外されます。

#### 列名の指定方法（次元・後処理・ファイル出力）

- **基本**
  - `eo`: 埋め込みベクトルをセルに JSON で出力（例: `[0.1,0.2,...]`）
  - `eof`: 埋め込みベクトルをファイルに出力し、セルにはファイル名（`output_dir` 相対）を書き込み
- **前段階（埋め込みAPIの標準機能）で次元指定**
  - `eon1536`: `dimensions=1536` を指定して埋め込み生成
- **後段階（全行の結果が出揃ってから）で次元削減**
  - `eop128`: PCA で 128 次元に削減
  - `eot2`: t-SNE で 2 次元に削減
  - `eou2`: UMAP で 2 次元に削減
- **前段 + 後段の併用**
  - `eon1536p128`: `dimensions=1536` → PCA で 128 次元
  - `eon1536p128f`: 上記 + ファイル出力

#### 出力形式（セル直書き / ファイル）

- **セル直書き（`f` なし）**: JSON 文字列でセルに書き込みます。
- **ファイル出力（`f` あり）**: 常にファイルに保存し、セルにはファイル名を書き込みます。
- **フォールバック**: `f` なしでも、セルの最大文字数（目安: 32767）を超える場合は自動的にファイル出力にフォールバックします。
- **ファイル形式/命名**:
  - 形式: NumPy `.npy`（float32）
  - 命名: `eof{index}_{row}.npy`（`index` は `eo*` 列の左→右の 1 始まり、`row` は DataFrame 行番号の 1 始まり）

#### 実行例

```bash
python -m dokabun -i examples/sample.csv --model openai/text-embedding-3-small
```

`examples/sample.csv` には最小構成のサンプルが含まれています。

| i_content | so_summary\|本文の要約 | so_sentiment\|感情ラベル |
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
