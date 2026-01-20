# dokabun

スプレッドシートをインターフェースとして LLM の構造化/非構造化出力を活用する CLI ツールです。  
`i_` で始まる列を分析対象として読み込み、空欄になっている出力列を OpenRouter 経由で自動的に埋めます。

## 使い方

### ① スプレッドシートを用意する

ヘッダーのプレフィックスを使って、入力やAIへの指示を指定します。

##### シート作成例

| i_content | so_summary\|本文の要約 | so_sentiment\|感情ラベル |
| --- | --- | --- |
| 今日はとても暑い一日でした。 | (空欄) | (空欄) |
| 新商品のレビューを集めています。 | (空欄) | (空欄) |

##### プレフィックス一覧

| プレフィックス | 説明 | 列名の入力例 | セル入力例と処理結果 | 注意点 |
| :--- | :--- | :--- | :--- | :--- |
| `i_` | **入力 (Input)**<br>分析対象のデータ | `i_content`<br>`i_画像ファイル` | `こんにちは` → (そのままテキストとして入力)<br>`docs/memo.txt` → (ファイル内容を展開して入力)<br>`images/image01.png` → (画像として入力) | 複数列ある場合は左から順に連結されます。<br>画像パスも自動認識されます。 |
| `so_` | **構造化出力**<br>指定した形式で抽出 | `so_summary\|要約`<br>`so_感情` | (空欄) → `{"summary": "この文章は..." }`<br>(空欄) → `{"感情": "positive"}` | `\|` の後ろに説明を書くと精度が上がります。<br>空欄のセルのみ埋められます。 |
| `nso_` | **テキスト出力(非構造化出力)**<br>自由記述で生成 | `nso_このテキストを翻訳してください`<br>`nso_この計画を評価してください。` | (空欄) → `Hello, world.` <br>(空欄) → `Good job!` | 列ごとに独立して生成されます。<br>列名の後ろ部分がプロンプトになります。 |
| `nsof_` | **ファイル出力(非構造化出力のファイル出力)**<br>生成結果を別ファイルに保存 | `nsof_code`<br>`nsof_log` | (空欄) → `nsof1_2.txt` (生成内容はファイルへ保存) | セルには保存されたファイル名が入ります。 |
| `eo` / `eof` | **埋め込み**<br>ベクトル化 | `eo`<br>`eof`<br>`eon1536t2` | (空欄) → `[0.1, 0.2, ...]` (JSON)<br>(空欄) → `eof1_2.npy` (NumPyファイル)<br>(空欄) → `[0.2, 0.3]` | `--model` に埋め込み対応モデルの指定が必要です。 |
| `l_` | **ラベル (Label)**<br>メタデータ | `l_id`<br>`l_note` | `ID-001`<br>`確認済み` | LLMの入力には含まれず、処理に影響しません。 |

詳細な仕様（ファイルパスの解決ルール、埋め込みの次元指定など）については、[スプレッドシート形式の詳細](#スプレッドシート形式の詳細)を参照してください。

### ② 実行する

**API キーの指定（推奨: 環境変数）**

```powershell
setx OPENROUTER_API_KEY "sk-..."
```

```bash
export OPENROUTER_API_KEY="sk-..."
```

`--api-key` でも指定できます（シェル履歴に残りやすいので注意）。

```bash
uvx dokabun --api-key "sk-..." -i examples/sample.csv --model openai/gpt-4.1-mini
```

**実行方法（いずれか）**

- **uvx（最もおすすめ。uvがあればインストール不要で実行可能！）**
  ```bash
  uvx --from git+https://github.com/Nu424/dokabun dokabun -i examples/sample.csv --model openai/gpt-4.1-mini --api-key "{OpenRouterのAPIキー}"
  # uvx dokabun -i examples/sample.csv --model openai/gpt-4.1-mini # こちらも可能
  ```
  ローカルのリポジトリから試す場合:
  ```bash
  uvx --from . dokabun -i examples/sample.csv --model openai/gpt-4.1-mini
  ```
  まずはヘルプ:
  ```bash
  uvx --from . dokabun --help
  ```

- **python -m（ローカル開発向け）**
  ```bash
  uv sync  # もしくは pip install -e .
  python -m dokabun -i examples/sample.csv --model openai/gpt-4.1-mini
  ```

- **uv run（プロジェクト内で実行）**
  ```bash
  uv sync
  uv run dokabun -i examples/sample.csv --model openai/gpt-4.1-mini
  ```
  `uv run python -m dokabun ...` でも実行できます。

#### OpenAI 互換エンドポイントに切り替える例

```bash
uvx dokabun -i examples/sample.csv --model gpt-4o-mini --base-url https://api.openai.com/v1
```

#### 埋め込み列を使う場合のモデル例

```bash
uvx dokabun -i examples/sample.csv --model openai/text-embedding-3-small
```

#### UMAP（`eou*`）を使う場合（追加インストール）

UMAP は `umap-learn`（内部で `numba/llvmlite` を利用）に依存するため、**任意機能**として切り出しています。

- `uv` でプロジェクト環境に入れる場合:

```bash
uv sync --extra umap
```

- `uvx` でその場実行する場合（ローカルリポジトリ）:

```bash
uvx --from . "dokabun[umap]" -i examples/sample.csv --model openai/text-embedding-3-small
```

#### 埋め込み可視化 CLI（追加インストール）

埋め込みベクトル列（セル内の `"[x,y]"` 形式、または `.npy` ファイル名）を Plotly で可視化します。

- `uv` でプロジェクト環境に入れる場合:

```bash
uv sync --extra viz
```

- `uvx` でGitHubから直接実行する場合:

```bash
uvx --from git+https://github.com/Nu424/dokabun[viz] dokabun-viz -i path/to/input_YYYYMMDD_HHMMSS.out.xlsx --embedding-col eot2 --label-col l_id
```

- `uvx` でその場実行する場合（ローカルリポジトリ）:

```bash
uvx --from ".[viz]" dokabun-viz -i path/to/input_YYYYMMDD_HHMMSS.out.xlsx --embedding-col eot2 --label-col l_id
```

- 補足:
  - 次元は自動判定され、2次元/3次元のみ対応（4次元以上はエラー）。
  - ホバー表示は `--hover-col` で指定可能。未指定時は `i_` の先頭列を使用。
  - ホバー表示文字数は `--hover-max-chars` で制限可能。

#### 主なオプション

| オプション | 説明 |
| --- | --- |
| `-i, --input` | 入力スプレッドシート (`.xlsx` または `.csv`) |
| `--api-key` | API キー（未指定時は環境変数を参照） |
| `--model` | 使用するモデル名 (既定: `openai/gpt-4.1-mini`) |
| `--base-url` | LLM API のベース URL (既定: `https://openrouter.ai/api/v1`) |
| `--concurrency` | 同時並列数 (既定: 5) |
| `--partial-interval` | 何行ごとに一時保存するか |
| `--max-text-file-bytes` | テキストファイル読み込み時の最大サイズ（バイト）(既定: 262144) |
| `--timestamp` | 途中再開したいタイムスタンプを明示指定 |
| `--nsof-ext` | `nsof_` 列の出力拡張子 (`txt` / `md`) |
| `--nsof-name-template` | `nsof_` 通常時のファイル名テンプレ |
| `--nsof-name-template-filetarget` | `i_` が1列かつファイルパス時のファイル名テンプレ |

### 実行結果

実行すると、`output_dir`（未指定時は入力ファイルと同じディレクトリ）に下記ファイルが出力されます。

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
    - `configure_logging(...)` / `args.api_key` or `_load_api_key()`
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

## スプレッドシート形式の詳細

### 入力列 (`i_`) の詳細仕様

- **テキストファイル**: 拡張子 `.txt`, `.md`, `.markdown`, `.log`, `.csv`, `.json`, `.yml`, `.yaml` を認識します。
- **画像ファイル**: 拡張子 `.png`, `.jpg`, `.jpeg`, `.webp` を認識します。
- **パス解決**: ファイルパスはスプレッドシートファイルからの相対パスとして解釈されます。
- **複数列**: 複数ある場合は左から順にプロンプトへ渡されます。

### 非構造化出力列 (`nso_`, `nsof_`) の詳細仕様

- `nsof_{プロンプト}` のファイル保存名ルール:
  - デフォルト名: `nsf{index}_{行番号}.txt`（行番号はデータ行 1 始まり）
    - 例: `nsof1_1.txt`, `nsof1_2.txt`, `nsof1_3.txt`, ...
  - `i_` 列が 1 つだけで、その値が実在するファイルパスの場合: `{target_file_stem}_nsf{index}.txt` を採用します。
    - 例: `text01.txt` -> `text01_nsof1.txt`

### 埋め込み出力列 (`eo*`) の詳細仕様

- **埋め込み入力**: `i_` 列のうち `TextTarget` になったものを左→右で連結（区切りは `\n\n`）してベクトル化します。画像は除外されます。

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

## テスト

主要コンポーネントには `pytest` ベースのテストが付属しています。

```bash
pytest
```

Fake クライアントを用いてネットワークアクセスなしで `process_row_async` を検証しています。

## ライセンス

MIT License
