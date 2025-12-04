### どかぶん フェーズ5〜7 実装計画

このドキュメントは、すでに実装済みのフェーズ1〜4（`config` / `logging_utils` / `io.spreadsheet` / `target` / `preprocess` / `llm`）を前提に、  
フェーズ5〜7をスムーズに実装するための詳細なガイドと「開発のポイント」をまとめたものです。  
全体像は既存の [`document/step.md`](document/step.md) を踏まえつつ、実際のモジュール構成に寄せて具体化しています。

---

### 1. 前提とゴール

- **前提**
  - すでに以下が利用可能：
    - 設定: `dokabun.config.AppConfig`
    - ロギング: `dokabun.logging_utils.configure_logging` / `get_logger`
    - スプレッドシート I/O: `dokabun.io.spreadsheet.SpreadsheetReaderWriter`
    - 分析対象表現: `dokabun.target.TextTarget` / `ImageTarget`
    - 前処理パイプライン: `dokabun.preprocess.run_preprocess_pipeline`
    - スキーマ・プロンプト: `dokabun.llm.schema.build_schema_from_headers`, `dokabun.llm.prompt.build_prompt`
    - LLM クライアント: `dokabun.llm.openrouter_client.AsyncOpenRouterClient`
- **フェーズ5〜7の最終ゴール**
  - `python -m dokabun -i input.xlsx` のようなコマンドで、
    - スプレッドシートの未入力セルを LLM 構造化出力で自動埋め込み
    - 処理中断からの再開・進捗表示・ログ出力・コストサマリー出力が行える状態にする。

開発を円滑に進めるため、**「1行処理ロジック → 並列実行制御 → CLI → サマリーとテスト」**の順で段階的に完成させる方針をとります。

---

### 2. フェーズ5: コア処理ランナーと非同期実行

フェーズ5では、実際に 1 行ずつスプレッドシートを処理し、LLM と連携して値を埋めていく**コアロジック**を実装します。

#### 2.1 モジュール構成と責務

- 追加するファイル:
  - `dokabun/core/runner.py`
- 主な責務:
  - スプレッドシートの読み込みと「どの行・どの列を埋めるか」の決定。
  - 1 行単位の処理ロジック（前処理 → スキーマ生成 → プロンプト生成 → LLM 呼び出し → DataFrame 書き戻し）。
  - `asyncio` を用いた並列実行と、`SpreadsheetReaderWriter` による一時保存・再開。

#### 2.2 行単位処理ロジック（同期版の骨組み）

1. **列の分類ヘルパーの実装**
   - `t_` で始まる列名を「分析対象列」として抽出。
   - それ以外を「出力候補列」とし、ヘッダ文字列（`{列名}` / `{列名}|{説明}`）をそのままスキーマ生成に渡せる形で保持。
   - ポイント:
     - 列順を保存する（`build_schema_from_headers` にそのまま渡すため）。
     - 将来的な型拡張に備え、型情報を別途付加できる構造にしておくと拡張が容易。

2. **行ごとの「処理対象出力セル」の決定**
   - 各行について、出力候補列のうち「セルが空（NaN または空文字列）」のものだけを今回の対象とする。
   - 対象セルが 1 つもない行はスキップする（LLM 呼び出しコスト削減）。

3. **行単位処理関数（同期版）の設計**
   - `def process_row_sync(row_index: int, df: pd.DataFrame, target_columns: list[str], output_columns: list[str], base_dir: Path, client: AsyncOpenRouterClient, config: AppConfig) -> RowResult:`
     - 中で行う処理:
       1. **入力の構築**
          - 行 `row = df.iloc[row_index]` から、`target_columns` の値を左から順に取り出す。
          - 各セル値を文字列化し、`run_preprocess_pipeline(value, base_dir)` に渡して `Target` を生成。
          - 生成した `Target` のリストを 1 行分の `targets` としてまとめる。
       2. **スキーマ生成**
          - 出力候補列のヘッダ（`df.columns` から `t_` 以外を取得）を `build_schema_from_headers` に渡して JSON Schema を構築。
       3. **プロンプト生成**
          - `build_prompt(row_index, row, targets, schema)` を呼び出し、`messages` と `response_format` を取得。
       4. **LLM 呼び出し（※この時点ではインターフェースのみ意識）**
          - 実装のしやすさのため、同期関数では「どのような引数を LLM クライアントに渡すか」だけを明確にしておき、実処理は後の非同期化で行う。
       5. **結果のパース & 書き戻し**
          - LLM 応答から `dict` を取り出し、対応する出力列に値を書き戻すロジックを整理しておく。
     - `RowResult`（後で `@dataclass` 化）には、少なくとも以下を含める想定:
       - `row_index`: 行インデックス
       - `updates`: `{列名: 値}` の辞書
       - `usage`: LLM 呼び出しのトークン・コスト情報
       - `error`: 異常時のエラー内容（成功時は `None`）

4. **開発のポイント**
   - まずは **LLM 呼び出しをダミー（スタブ関数）にして**、前処理〜スキーマ〜プロンプト〜書き戻しの流れだけをローカルで確認するとデバッグが楽です。
   - 行単位処理は「状態を持たない純粋関数」に近づけるほどテストしやすく、後の非同期化もスムーズになります。

#### 2.3 非同期化と `asyncio.Semaphore` による並列制御

1. **非同期版行処理関数の実装**
   - `async def process_row_async(..., client: AsyncOpenRouterClient, ...) -> RowResult:`
   - フェーズ 2.2 で整理した同期版ロジックをベースに、LLM 呼び出し部分だけを `await client.create_completion(...)` に置き換える。
   - `client.create_completion` には、`AppConfig.temperature` / `AppConfig.max_tokens` をそのまま渡す（`None` の場合はモデル側のデフォルト）。

2. **全体実行関数の設計**
   - `async def run_async(config: AppConfig) -> None:`
     - ステップ:
       1. `SpreadsheetReaderWriter(config)` を初期化し、`df = reader.load()` で DataFrame を取得。
       2. `reader.load_meta_if_exists()` を呼んでメタデータがあれば読み込み、`start_row_index` を決定（例: `last_completed_row + 1`）。
       3. スキップ対象行（すべての出力セルが埋まっている行）を除いた「未処理行インデックス」のリストを作成。
       4. `AsyncOpenRouterClient` を初期化（API キーやモデル名、タイムアウトは `AppConfig` と環境変数から）。
       5. `asyncio.Semaphore(config.max_concurrency)` を用意し、各行処理タスクで `async with semaphore:` する。
       6. 行インデックスごとに `process_row_async` をタスク化し、`asyncio.as_completed` + `tqdm` で進捗表示しながら結果を回収。
       7. 一定件数（`config.partial_interval`）ごとに `SpreadsheetReaderWriter.save_partial(df, start_row, last_row)` を呼び出して一時保存。
       8. 全タスク完了後、`save_output(df)` で最終出力を保存。

3. **同期版ラッパ関数**
   - `def run(config: AppConfig) -> None:`
     - `asyncio.run(run_async(config))` を呼ぶだけの薄いラッパにし、フェーズ6の CLI からはこちらを利用する。

4. **途中中断と例外処理**
   - `run` で `try: ... except KeyboardInterrupt:` をキャッチし、
     - 直近の処理済み行インデックスを記録したメタデータを保存。
     - 「ユーザー中断」の旨を標準出力に表示。
   - 行レベルの例外（前処理失敗 / LLM 呼び出し失敗など）は `RowResult.error` に格納しつつ、他行の処理は継続する。

5. **開発のポイント**
   - DataFrame への書き戻しは **1 箇所に集約**する（例: `run_async` 内の結果回収ループ）。  
     複数タスクから同時に DataFrame を書き換えないことで、デバッグ性と一貫性が向上します。
   - `RowResult` に usage 情報（`prompt_tokens`, `completion_tokens`, `cost` など）を格納しておくと、フェーズ7のサマリー実装がスムーズになります。

---

### 3. フェーズ6: CLI エントリポイントとユーザー体験

フェーズ6では、エンドユーザーが `python -m dokabun` やコマンド名だけで直感的に使えるように、CLI を整備します。

#### 3.1 モジュール構成

- 追加・更新するファイル:
  - `dokabun/cli.py`（新規）
  - `dokabun/__main__.py`（既存の暫定実装から本実装へ差し替え）

#### 3.2 CLI の設計

1. **引数パーサの実装**
   - `argparse` を用いて `build_parser()` / `parse_args()` を実装。
   - 想定オプション:
     - `--input, -i`（必須）: 入力スプレッドシートパス → `AppConfig.input_path`
     - `--model`: モデル名（デフォルト `openai/gpt-4.1-mini`）
     - `--temperature`: 温度（`float`、省略時は `None` → モデルデフォルトを利用）
     - `--max-tokens`: 最大トークン数
     - `--concurrency`: 同時並列数 → `AppConfig.max_concurrency`
     - `--max-rows`: 実行あたりの最大処理行数（デバッグ用途）
     - `--partial-interval`: 一時保存間隔（行数）→ `AppConfig.partial_interval`
     - `--output-dir`: 出力ディレクトリ → `AppConfig.output_dir`
     - `--log-level`: ログレベル（`INFO` / `DEBUG` / `WARNING` / `ERROR`）
     - （オプション）`--dry-run`: LLM を呼ばずに「どの行が対象になるか」だけ表示するモード

2. **`AppConfig` 生成とロギング初期化**
   - `def main(argv: list[str] | None = None) -> int:` の中で:
     1. 引数をパースして `config_dict` を作成。
     2. `AppConfig.from_dict(config_dict)` で設定オブジェクトを生成。
     3. `configure_logging(config.log_level, config.log_file)` を呼んでログを初期化。

3. **OpenRouter API キーの取得と検証**
   - 環境変数 `OPENROUTER_API_KEY` から API キーを取得（必要に応じて別名も許容）。
   - 未設定の場合:
     - エラーメッセージを標準エラーに表示し、非ゼロ終了コードで終了。
   - API キーは `core.runner.run` に渡すか、`AsyncOpenRouterClient` 生成部分で直接参照するか、どちらかの方針を明示しておく。

4. **ランナー呼び出しと終了コード**
   - `core.runner.run(config, api_key=...)` のような形で呼び出し（`api_key` を引数に含める設計にするとテストもしやすい）。
   - 例外発生時はログにエラーを出力しつつ、CLI としては非ゼロコードを返す。

5. **`__main__.py` の更新**
   - 既存の暫定実装を削除し、`from .cli import main` を呼び出すだけの実装に変更。
   - `if __name__ == "__main__": raise SystemExit(main())` の形にしておくと、終了コードの扱いがわかりやすくなります。

6. **開発のポイント**
   - CLI は「**失敗したときに何が悪いのかがすぐわかる**」ことが重要です。
     - 入力ファイル未存在、API キー未設定、スプレッドシート形式の想定外など、代表的な失敗パターンごとに分かりやすいメッセージを用意しておくと、後の運用が楽になります。
   - ログレベル `DEBUG` のときだけ詳細なスタックトレースを出すようにしておくと、開発と本番での使い分けがしやすくなります。

---

### 4. フェーズ7: ログ・サマリー・テスト・チューニング

フェーズ7では、これまでの実装を**安心して回せるバッチツール**に仕上げるための仕上げを行います。

#### 4.1 サマリー集計とレポート

1. **`core.summary.ExecutionSummary` の設計**
   - ファイル: `dokabun/core/summary.py`
   - フィールド例:
     - `started_at`, `finished_at`
     - `total_rows`, `processed_rows`, `success_rows`, `failed_rows`
     - `prompt_tokens`, `completion_tokens`, `total_tokens`, `total_cost_usd`
     - `error_counts`: `{エラー種別: 件数}`
   - メソッド例:
     - `start()` / `finish()`
     - `record_success(row_index: int, usage: dict[str, Any])`
     - `record_failure(row_index: int, error_type: str, message: str)`
     - `format_text()`（人間向けテキストサマリー）
     - （オプション）`to_dict()` / `to_json()`（機械可読なサマリー出力）

2. **ランナーとの連携**
   - `run_async` の開始時に `ExecutionSummary` を初期化し、行処理結果（`RowResult`）ごとに `record_success` / `record_failure` を呼び出す。
   - `AsyncOpenRouterClient` のレスポンスから `usage` 情報を取り出し、サマリーに加算。
   - 終了時に `summary.format_text()` を標準出力に表示し、同時にログにも INFO レベルで記録する。

3. **開発のポイント**
   - サマリーは「1 回の実行を振り返るためのダッシュボード」として設計すると、集計項目が自然と決まります。
   - 将来的に JSON 形式で機械集計したくなる可能性が高いため、`to_dict()` のような内部表現を最初から用意しておくと拡張しやすくなります。

#### 4.2 テスト戦略

1. **ユニットテスト対象**
   - すでにあるフェーズ1〜4のモジュールも含めて、以下を中心にテスト:
     - `build_schema_from_headers`（ヘッダ → JSON Schema）
     - `run_preprocess_pipeline`（プレーンテキスト / 画像パスの判定と変換）
     - `SpreadsheetReaderWriter`（一時ディレクトリを使った読み書き・メタデータ保存／読込）
     - `AppConfig`（パスの正規化とバリデーション）

2. **ランナーのテスト**
   - ネットワーク呼び出しを行わないよう、`AsyncOpenRouterClient` をモック化する。
     - 例: `FakeOpenRouterClient` をテスト用に用意し、`create_completion` が固定の JSON を返すようにする。
   - 小さな DataFrame を直接作成し、「2〜3 行だけ処理する」形で `process_row_async` / `run_async` をテスト。

3. **CLI のテスト**
   - `argparse` のパース結果を直接検証するユニットテスト。
   - `subprocess` で `python -m dokabun` を実行し、ヘルプ出力と代表的なエラーケース（入力ファイルなし / API キー未設定）を確認する軽い統合テスト。

4. **開発のポイント**
   - LLM 呼び出しを含むテストはコストがかかるため、「**本物の API を叩くテスト**」はごく小さなケースを 1〜2 個に限定し、手動または低頻度の自動テストに留めます。
   - それ以外は極力モックでカバーし、テストを高速・安定に保つことを優先します。

#### 4.3 パフォーマンス・運用チューニング

1. **並列数と一時保存間隔のチューニング**
   - `max_concurrency` と `partial_interval` は入力データサイズやネットワーク状況に応じて調整が必要。
   - 実運用前に、小さなサンプルで以下を確認:
     - あまりにも並列数を上げすぎると OpenRouter のレートリミットにかかる。
     - 一時保存間隔が短すぎると I/O オーバーヘッドが増える。

2. **ログの粒度調整**
   - デフォルト `INFO` では「ファイル開始・終了」「一時保存」「サマリー」程度に留める。
   - `DEBUG` では 1 行ごとの詳細な処理内容（`RowResult` など）をログに出せるようにする。

3. **将来の拡張を見据えた余地**
   - URL / PDF 対応を追加する際は、`Preprocess` 実装と `Target` 型を追加するだけで済むよう、ランナー側は「Target のリスト」として一般化した構造にしておく。
   - 出力先を DB や JSONL に変えたくなった場合に備え、`SpreadsheetReaderWriter` と同様のインターフェースを持つ別 Writer を差し替えられるよう設計しておく。

---

このフェーズ5〜7実装計画に沿って開発を進めることで、既存のフェーズ1〜4の実装を最大限活かしつつ、  
エンドユーザーが安心して使える「どかぶん」CLI を段階的に完成させることができます。***

