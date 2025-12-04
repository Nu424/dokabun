### どかぶん 実装ステップ

このドキュメントは、`dokabun` パッケージを設計書どおりに実装していくための、具体的なステップ一覧です。  
基本的には上から順に進めれば、最小動作する CLI ツール → 機能拡張 の順で完成していきます。

> 重要: 各フェーズが実装完了次第、そのフェーズの実装結果・実装内容などを、このドキュメントの各所に追記すること

---

### フェーズ 0: 開発環境・プロジェクトの準備

1. **Python 環境と依存パッケージの準備**
   - `uv` で以下をインストール：`pandas`, `openpyxl`, `tqdm`, `openai`, `python-dateutil`, `pytest`（必要に応じて）。
2. **パッケージ構造の作成**
   - プロジェクトルートに `dokabun/` ディレクトリを作成し、`__init__.py` を追加。
   - `pyproject.toml` に `dokabun` をパッケージとして登録し、エントリポイント（後で `python -m dokabun` で動かす想定）を確認。

---

### フェーズ 1: 設定とロギングの土台づくり

3. **設定クラス `config.AppConfig` の実装**
   - `dokabun/config.py` を作成。
   - 以下の項目を持つ設定クラスを定義（`dataclasses` もしくは `pydantic`）：  
     - 入力ファイルパス、出力ディレクトリ（省略時は入力と同じ）、タイムスタンプ付きファイル命名設定。
     - 使用モデル名（例：`openai/gpt-4.1-mini`）、`temperature`、`max_tokens`。
     - 同時並列数、1 度の実行で処理する最大行数（オプション）、一時ファイルの出力間隔（N 行ごと）。
   - CLI 引数（次フェーズで実装）から `AppConfig` を生成できるようにしておく。
4. **ロギング共通処理 `logging_utils` の実装**
   - `dokabun/logging_utils.py` を作成。
   - ログレベル（INFO/ERROR/DEBUG 等）とログファイル出力（例：`dokabun.log`）を設定する関数 `get_logger(name: str)` を用意。

---

### フェーズ 2: スプレッドシート I/O とタイムスタンプ付きファイル

5. **タイムスタンプ生成ユーティリティの実装**
   - 共通ユーティリティとして、`YYYYMMDD_hhmmss` 形式のタイムスタンプ文字列を返す関数を実装（例：`utils.datetime.ts_now()` など、好みで場所を決める）。
6. **`io.spreadsheet.SpreadsheetReaderWriter` の設計・実装**
   - `dokabun/io/spreadsheet.py` を作成。
   - 主な責務：
     - 入力ファイルの存在チェックと読み込み（`pandas.read_excel` / `read_csv`）。
     - 実行開始時にタイムスタンプ `TS` を生成し、  
       - `input_TS.xlsx`（オリジナルのコピー）  
       - `input_TS.out.xlsx`（最終出力用）  
       を扱うパスを決定する。
     - DataFrame と、元ファイルパス／出力ファイルパスの管理。
7. **一時ファイル出力と再開メタデータの枠組み**
   - `SpreadsheetReaderWriter` に以下のメソッドを追加：
     - `save_partial(df, start_row, end_row)`：`input_TS.partial.<start>-<end>.xlsx` を出力。
     - 再開時に使用するメタデータ（最後に完了した行インデックスなど）を簡易に JSON などで保存・読込。
   - 再開ロジック自体は後続フェーズ（`core.runner`）で利用する前提の「I/O 部分の土台」として実装。

---

### フェーズ 3: 分析対象表現と前処理

8. **ターゲット表現 `target.py` の実装**
   - `dokabun/target.py` を作成。
   - `TextTarget` / `ImageTarget` クラス（もしくは `@dataclass`）を定義し、共通インターフェース `to_llm_content()` を持たせる。
     - `TextTarget`：`{"type": "text", "text": ...}` を返す。
     - `ImageTarget`：`{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` 形式を返す（OpenRouter の仕様に合わせる）。
9. **前処理抽象クラス `Preprocess` の実装**
   - `dokabun/preprocess/base.py` を作成。
   - インターフェース：
     - `is_eligible(target_text: str) -> bool`
     - `preprocess(target_text: str, base_dir: Path) -> Target`
10. **具体前処理 `PlainTextPreprocess` / `ImagePreprocess` の実装**
    - `dokabun/preprocess/text.py` に `PlainTextPreprocess` を実装。
      - どの文字列も受け入れ、軽い整形（strip, 改行整形）を行って `TextTarget` を返す。
    - `dokabun/preprocess/image.py` に `ImagePreprocess` を実装。
      - 拡張子チェック（`.png`, `.jpg`, `.jpeg`, `.webp`）。
      - スプレッドシートファイルのディレクトリを `base_dir` として相対パス→絶対パスへ解決し、ファイル存在チェック。
      - 画像を読み込み、Base64 へ変換して `ImageTarget` を返す。
11. **前処理パイプラインのヘルパー実装**
    - 前処理クラスのリスト（`[ImagePreprocess(), PlainTextPreprocess()]` 等）を保持し、`t_xxx` のセル値ごとに
      - 最初に `is_eligible == True` となったものを選び `preprocess` を実行する関数を `preprocess/__init__.py` などに用意。

---

### フェーズ 4: スキーマ生成・プロンプト・OpenRouter クライアント

12. **スキーマ生成 `llm.schema.build_schema_from_headers` の実装**
    - `dokabun/llm/schema.py` を作成。
    - 出力対象列のヘッダ（`{列名}` / `{列名}|{説明}`）から JSON Schema を構築：
      - `properties` に `{"列名": {"type": "string", "description": "説明"}}` を追加。
      - `required` に全ての列名を追加（全項目必須）。
13. **プロンプト生成 `llm.prompt.build_prompt` の実装**
    - `dokabun/llm/prompt.py` を作成。
    - 入力：
      - 行データ（`pandas.Series` など）、ターゲット群（`TextTarget`/`ImageTarget`）、スキーマ。
    - 出力：
      - OpenAI SDK 用の `messages` 配列（`system` / `user` など）を返す。
      - `user` メッセージには、結合した `t_xxx` 内容や必要なメタ情報を含める。
14. **OpenRouter クライアント `AsyncOpenRouterClient` の実装**
    - `dokabun/llm/openrouter_client.py` を作成。
    - OpenAI SDK のクライアントをラップするクラスを定義：
      - 初期化時に `base_url="https://openrouter.ai/api/v1"` と `api_key` を受け取る。
      - メソッド `async def create_completion(...)` で
        - `model`, `messages`, `response_format`（`json_schema`）、`temperature`, `max_tokens`, `usage.include=True` を指定し、レスポンス JSON を返す。
      - タイムアウトやリトライは内部で扱うか、呼び出し元から指定できるようにする。

---

### フェーズ 5: コア処理ランナーと非同期実行

15. **行単位処理ロジックの実装（同期版の骨組み）**
    - `dokabun/core/runner.py` を作成。
    - まずは非同期化せず、1 行だけ処理する関数を実装：
      - 入力：設定、DataFrame、行インデックス。
      - 処理：
        1. `t_xxx` 列から対象セルを集めて前処理 → `Target` 群へ。
        2. 出力列からスキーマを構築。
        3. プロンプト生成。
        4. OpenRouter クライアントで構造化出力を取得。
        5. JSON から値を取り出し、対応するセルに書き戻す。
16. **非同期化と `asyncio.Semaphore` による並列制御**
    - 上記ロジックを `async def process_row_async(...)` にリファクタリング。
    - 同ファイル内に以下を実装：
      - `async def run_async(config: AppConfig)`:  
        - スプレッドシート読み込み、未処理行の抽出、`asyncio.Semaphore` で並列数を制御しつつ `process_row_async` を `gather`。
        - 一定行数ごとに一時保存メソッドを呼び出し。
      - `def run(config: AppConfig)`:  
        - `asyncio.run(run_async(config))` で実行。
17. **途中中断と一時ファイル再開の実装**
    - `KeyboardInterrupt` キャッチや例外処理を追加。
    - 一時ファイルに記録した「最終完了行インデックス」を読み取り、次回起動時にはそこから再開するロジックを `run_async` に組み込む。

---

### フェーズ 6: CLI エントリポイントとユーザー体験

18. **CLI エントリ `cli.py` の実装**
    - `dokabun/cli.py` を作成。
    - `argparse` などを使って以下のオプションを定義：
      - `--input` / `-i`: 入力スプレッドシートパス（必須）。
      - `--model`: モデル名（省略時はデフォルト）。
      - `--concurrency`: 同時並列数。
      - `--partial-interval`: 一時保存間隔（行数）。
      - `--output-dir`: 出力ディレクトリ（省略時は入力と同じ）。
      - `--log-level`: ログレベル。
    - 引数から `AppConfig` を生成し、`core.runner.run(config)` を呼び出す。
19. **`python -m dokabun` で動くようにする**
    - `dokabun/__main__.py` を作成し、`from .cli import main` を呼んで CLI を起動。
    - 必要に応じて `pyproject.toml` の `[project.scripts]` でエントリポイントも追加。

---

### フェーズ 7: ログ・サマリー・テスト・チューニング

20. **サマリー集計 `core.summary.ExecutionSummary` の実装**
    - 成功行数・失敗行数・トークン数・推定コスト・実行時間などを集計するクラスを実装。
    - `runner` から呼び出し、処理終了時に人間向けのテキストサマリーを標準出力へ出す。
21. **基本的なユニットテストの追加**
    - `tests/` ディレクトリを作成。
    - 以下を中心にテストを書く：
      - スキーマ生成（ヘッダ→JSON Schema）。
      - 前処理（プレーンテキスト／画像パスの判定と変換）。
      - スプレッドシート I/O（小さなサンプルファイルを使った読み書き）。
22. **サンプルスプレッドシートとドキュメントの用意**
    - `examples/` ディレクトリなどに、最小の Excel/CSV サンプルを配置。
    - `README.md` もしくは別ドキュメントに、「サンプルファイルを使った実行例（コマンド例 / 期待される出力）」を追記。

---

このステップに沿って実装を進めることで、まずは **テキスト＋画像に対応した最小のどかぶん CLI** を完成させ、その後に URL/PDF 対応や追加オプションなどを段階的に拡張していくことができます。

