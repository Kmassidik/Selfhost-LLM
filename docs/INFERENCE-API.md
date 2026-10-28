# Glicc Inference API

OpenAI-compatible endpoint for the self-hosted Qwen3.8 MoE model.

```text
Base URL: https://llm.glicc.id/v1
Model:    qwen3.8-moe
Context:  262,144 tokens
MTP:      enabled (depth 2)
Slots:    1
```

## API keys

Create per-client keys in the UI:

```text
https://llminferences.glicc.id/keys
```

Sign in, enter a client name and RPM, then copy the key immediately. It is shown once; only its SHA-256 hash is stored.

Revoke a key from the same page. Revocation is immediate for new requests.

The public inference host is the queue gateway:

```text
llm.glicc.id -> 127.0.0.1:8081 gateway
                  -> 127.0.0.1:8084 llama.cpp
```

The llama.cpp global key is now backend-only and no longer works on the public endpoint.

CLI fallback:

```bash
cd /root/Desktop/gpu-optimized/gateway
.venv/bin/python manage_keys.py issue alice --rpm 60
.venv/bin/python manage_keys.py list
.venv/bin/python manage_keys.py revoke alice
```

Do not put a key in a repository. Use an environment variable or local secret store.

## Quick test

```bash
curl https://llm.glicc.id/v1/chat/completions \
  -H "Authorization: Bearer $GLICC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-moe",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

## OpenAI Python SDK

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.glicc.id/v1",
    api_key=os.environ["GLICC_API_KEY"],
)

stream = client.chat.completions.create(
    model="qwen3.8-moe",
    messages=[{"role": "user", "content": "Explain KV cache in 3 bullets."}],
    max_tokens=512,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## oh-my-pi / omp

Profile path:

```text
~/.omp/profiles/glicc/agent/models.yml
```

```yaml
providers:
  glicc:
    baseUrl: https://llm.glicc.id/v1
    api: openai-completions
    apiKey: GLICC_API_KEY
    authHeader: true
    models:
      - id: qwen3.8-moe
        name: Glicc Qwen3.8 MoE
        reasoning: true
        supportsTools: true
        input: [text, image]
        contextWindow: 256000
        maxContextWindow: 256000
        maxTokens: 8192
        cost:
          input: 0
          output: 0
          cacheRead: 0
          cacheWrite: 0
        compat:
          supportsReasoningEffort: true
          thinkingFormat: qwen-chat-template
          qwenTemplateReasoningEffort: true
          reasoningContentField: reasoning_content
```

`~/.omp/profiles/glicc/agent/config.yml`:

```yaml
modelRoles:
  default: glicc/qwen3.8-moe
defaultThinkingLevel: auto
```

Export the key before starting:

```bash
export GLICC_API_KEY="$(cat /path/to/glicc-api.token)"
glicc
```

## OpenCode

OpenCode uses the AI SDK OpenAI-compatible provider. Put this in:

```text
~/.config/opencode/opencode.json
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glicc": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Glicc",
      "options": {
        "baseURL": "https://llm.glicc.id/v1",
        "apiKey": "{env:GLICC_API_KEY}"
      },
      "models": {
        "qwen3.8-moe": {
          "name": "Qwen3.8 MoE 256K",
          "limit": { "context": 256000, "output": 8192 },
          "reasoning": true,
          "tool_call": true,
          "attachment": true
        }
      }
    }
  }
}
```

```bash
export GLICC_API_KEY="$(cat /path/to/glicc-api.token)"
opencode
```

Select `glicc/qwen3.8-moe` in the model picker.

## Other harnesses

Any harness with an OpenAI-compatible base URL can use:

```text
Base URL: https://llm.glicc.id/v1
API key:  $GLICC_API_KEY
Model:    qwen3.8-moe
Context:  262144
Streaming: yes
Tools:    yes
Vision:   yes (Qwen3.8 mmproj)
```

Examples: Continue, Cline, Roo Code, Kilo Code, Aider, LiteLLM, LangChain, LlamaIndex, Open WebUI.

Claude Code speaks Anthropic Messages, not OpenAI Chat Completions. Put LiteLLM or another OpenAI→Anthropic bridge in front if you need it.

## Behaviour

- One request slot; a second request queues.
- Long context is shared: one 250K request consumes the window.
- MTP drafts two tokens and verifies them.
- `temperature` 0.6, `top_p` 0.95, `top_k` 20 is the tuned profile. Greedy decoding can cause repetition on long reasoning.
- Reasoning is returned in `reasoning_content`.

## Errors

```text
401  missing/invalid API key
404  unknown model or endpoint
429  queue/rate limit (gateway mode)
502  backend unavailable
504  generation or queue timeout
```
