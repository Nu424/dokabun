# LLMAPI の使い方

## 基本
- LLMAPI へのリクエストは **OpenRouter** を介して行います。  
  - OpenRouter は、複数のプロバイダー API を **共通の呼び出し形式** で扱えるラッパーサービスです。
- もっともシンプルな呼び出し例は次のとおりです。

```typescript
const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: '<モデル名>',
    messages: [
        { role: 'system', content: '<システムプロンプト>' },
        { role: 'user', content: '<ユーザープロンプト>' },
        { role: 'assistant', content: '<アシスタントの返答>' },
        { role: 'user', content: '<ユーザーの入力>' },
        ...
    ],
    temperature: <温度>,
    max_tokens: <最大トークン数>,
    stream: <ストリーミングをオンにするかどうか>,
  }),
});
```

- レスポンスの典型例は以下のとおりです。

```json
{
  "id": "<ID>",
  "created": <作成日時>,
  "model": "<モデル名>",
  "usage": {
    "prompt_tokens": <入力トークン数>,
    "completion_tokens": <出力トークン数>,
    "total_tokens": <合計トークン数>
  },
  "choices": [
    {
      "finish_reason": "<終了理由>",
      "message": {
        "role": "<ロール>",
        "content": "<コンテンツ>",
        "tool_calls": [
          {
            "id": "<ID>",
            "type": "<型>",
            "function": {
              "name": "<関数名>",
              "arguments": "<引数>"
            }
          },
          ...
        ]
      },
    }
  ]
}
```

- 実際のコストを確認したい場合は、`body.usage.include` を **true** にします。  
  サンプルコードは次のとおりです。

```typescript
const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: '<モデル名>',
    messages: [...],
    usage: {
        include: true,
    },
  }),
});
```

- コスト付きレスポンス例:

```json
{
  "id": "<ID>",
  ...,
  "usage": {
      "prompt_tokens": <入力トークン数>,
      "completion_tokens": <出力トークン数>,
      "total_tokens": <合計トークン数>,
      "cost": <呼び出しにかかるコスト(USD)>
  }
}
```

## モデルの選び方
- 現在サポートされている主なモデルを以下に示します。`<モデル名>` を指定するだけで利用できます。

  - **OpenAI**
    - `openai/gpt-4.1`  
      1.05M コンテキスト, 最大 33K トークン, $2 / 入力 1M トークン, $8 / 出力 1M トークン
    - `openai/gpt-4.1-mini`  
      1.05M コンテキスト, 最大 33K トークン, $0.40 / 入力 1M トークン, $1.60 / 出力 1M トークン
    - `openai/gpt-4o-mini`  
      128K コンテキスト, 最大 16K トークン, $0.15 / 入力 1M トークン, $0.60 / 出力 1M トークン
    - `openai/o3`  
      推論モデル。200K コンテキスト, 最大 100K トークン, $2 / 入力 1M トークン, $8 / 出力 1M トークン
    - `openai/o4-mini`  
      推論モデル。200K コンテキスト, 最大 100K トークン, $1.10 / 入力 1M トークン, $4.40 / 出力 1M トークン

  - **Google**
    - `google/gemini-2.5-flash`  
      1.05M コンテキスト, 最大 66K トークン, $0.30 / 入力 1M トークン, $2.50 / 出力 1M トークン
    - `google/gemini-2.5-pro`  
      1.05M コンテキスト, 最大 66K トークン, $1.25 / 入力 1M トークン, $10 / 出力 1M トークン
    - `google/gemini-2.5-flash-lite-preview-06-17`  
      1.05M コンテキスト, 最大 66K トークン, $0.10 / 入力 1M トークン, $0.40 / 出力 1M トークン

  - **Anthropic**
    - `anthropic/claude-sonnet-4`  
      200K コンテキスト, 最大 64K トークン, $3 / 入力 1M トークン, $15 / 出力 1M トークン

## ストリーミング
- `stream: true` を指定すると、モデルの出力を **逐次** 受け取ることができます。
- 具体例は以下のとおりです。

```typescript
const question = 'How would you build the tallest building ever?';
const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4o',
    messages: [{ role: 'user', content: question }],
    stream: true,
  }),
});

const reader = response.body?.getReader();
if (!reader) {
  throw new Error('Response body is not readable');
}

const decoder = new TextDecoder();
let buffer = '';

try {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Append new chunk to buffer
    buffer += decoder.decode(value, { stream: true });

    // Process complete lines from buffer
    while (true) {
      const lineEnd = buffer.indexOf('\n');
      if (lineEnd === -1) break;

      const line = buffer.slice(0, lineEnd).trim();
      buffer = buffer.slice(lineEnd + 1);

      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') break;

        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices[0].delta.content;
          if (content) {
            console.log(content);
          }
        } catch (e) {
          // Ignore invalid JSON
        }
      }
    }
  }
} finally {
  reader.cancel();
}
```

## 画像の読み込み
- 画像は **URL** または **Base64** エンコードされたデータで指定します。  
  - Base64 の場合は `data:image/<フォーマット>;base64,<データ>` 形式です。
- 対応フォーマット: `image/png`, `image/jpeg`, `image/webp`
- リクエスト例:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<テキスト>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "<URLまたはBase64エンコードされたデータ>"
          }
        }
      ]
    }
  ]
}
```

## PDF の読み込み
- PDF は **Base64** で渡します。
- 解析方法は 3 種類:
  1. `native`   — LLM が標準サポートしている場合に使用。
  2. `pdf-text` — PDF をテキスト化してそのまま渡す。無料。
  3. `mistral-ocr` — OCR ツールで解析。複雑な PDF 向け。1000 ページあたり $2。
- OpenRouter は `native` が失敗すると `mistral-ocr` にフォールバックします。
- サンプルコード:

```typescript
function encodePdfToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      resolve(dataUrl.split(",")[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

const base64Pdf = await encodePdfToBase64(pdfFile);

const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4.1',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'このPDFの内容を要約してください。' },
          {
            type: 'file',
            file: {
              filename: 'sample.pdf',
              file_data: base64Pdf
            }
          }
        ]
      },
    ],
    plugins: [
      {
        id: "file-parser",
        pdf: {
          engine: "native" // native, pdf-text, mistral-ocr
        }
      }
    ]
  }),
});
```

### ファイル注釈で再解析を省力化
- 1 回目の解析で得た `annotations` を再利用すれば、2 回目以降のコストを削減できます。

```typescript
// ---1 回目のレスポンスを取得
const responseJson = await response.json();
const fileAnnotation = responseJson.choices[0].message.annotations; // 実際は適切に存在確認する

// ---2 回目の呼び出し
const response2 = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4.1',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'このPDFの内容を要約してください。' },
          {
            type: 'file',
            file: {
              filename: 'sample.pdf',
              file_data: "<base64エンコードされたPDFデータ>"
            }
          }
        ]
      },
      {
        role: 'assistant',
        content: "このPDFは、以下のような内容です。...",
        annotations: fileAnnotation // ← ここで annotations を指定
      },
      {
        role: 'user',
        content: "このPDFで、特に重要な部分を教えてください。"
      }
    ]
    // 以前の解析結果を使うため plugins は不要
  }),
});
```

## 構造化された出力
- **JSON Schema** を指定すると、モデルに構造化出力を強制できます。
- 例:

```typescript
const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4.1',
    messages: [
      { "role": "system", "content": "与えられたテキストから情報を抽出してください" },
      { "role": "user", "content": "田中太郎24歳。東京に住んでいる。好きな食べ物はカレー、ピザ、ハンバーグです。" }
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "person",
        strict: true,
        schema: {
          type: "object",
          properties: {
            name: { type: "string", description: "名前" },
            age: { type: "number", description: "年齢" },
            city: { type: "string", description: "住んでいる都市" },
            favoriteFoods: { type: "array", items: { type: "string" }, description: "好きな食べ物" }
          },
          required: ["name", "age", "city", "favoriteFoods"],
          additionalProperties: false
        }
      }
    }
  }),
});

const responseJson = await response.json();
const person = responseJson.choices[0].message.content;
console.log(person);
// 出力例
// {
//   "name": "田中太郎",
//   "age": 24,
//   "city": "東京",
//   "favoriteFoods": ["カレー", "ピザ", "ハンバーグ"]
// }
```

## ツール呼び出し
- LLM に外部ツールを使わせると、さらに多様な処理が可能です。
- ツールは **function 形式の JSON Schema** で定義します。
- フローは大きく 3 ステップです。
  1. ツール定義とユーザーメッセージを LLM に送信し、**どのツールを呼ぶか** を選択させる。
  2. モデルが返した `tool_calls` を実際のツール (例: REST API) に伝え、結果を取得。
  3. 取得した結果を `tool` ロールのメッセージとして LLM に送り、最終回答を生成させる。

```typescript
// ---メッセージ・ツールの定義
const messages = [
  { "role": "user", "content": "明日の仙台の天気を教えて" }
];
const tools = [
  {
    type: "function",
    function: {
      name: "getWeather",
      description: "APIを使用して、特定の天気予報を取得する",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "天気を取得する都市" },
          date: { type: "string", description: "天気を取得する日付" }
        },
        required: ["city", "date"],
      }
    }
  }
];

// ---ツールの選択
const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4.1',
    messages: messages,
    tools: tools
  }),
});

// ---ツールの呼び出し
const responseJson = await response.json();
const message = responseJson.choices[0].message;
for (const toolCall of message.tool_calls) {
  const toolName = toolCall.function.name;
  const { city, date } = toolCall.function.arguments;
  const toolResponse = await searchWeather(city, date); // あらかじめツールを定義しておく

  // ---ツール結果をメッセージに追加
  messages.push({
    "role": "tool",
    "toolCallId": toolCall.id,
    "name": toolName,
    "content": JSON.stringify(toolResponse)
  });
}

// ---ツール結果を LLM に渡す
const response02 = await fetch('https://openrouter.ai/api/v1/chat/completions', {
  method: 'POST',
  headers: {
    Authorization: `Bearer <OPENROUTER_API_KEY>`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'openai/gpt-4.1',
    messages: messages,
    tools: tools
  }),
});
```
