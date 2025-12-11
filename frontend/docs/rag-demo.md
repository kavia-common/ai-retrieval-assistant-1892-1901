# RAG Demo (ai-rag-lite)

This frontend integrates the local `ai-rag-lite` library and exposes a demo at `/rag-demo`.

## Run

1) From `frontend/`:
   - npm install
   - npm start
2) Open http://localhost:3000/rag-demo

## Features

- Ingest freeform text (split multiple documents by a blank line).
- Controls for chunk size and overlap.
- Provider selector:
  - Custom (default): no network calls. Uses a deterministic toy embedding and simple echo LLM.
  - OpenAI: uses OpenAI embeddings and Chat if a key is provided.
  - HuggingFace: uses HF embeddings when a token is provided. LLM remains Custom.
- MemoryVectorStore only (browser-friendly).
- Shows top matches and the generated answer.

## Environment variables (optional)

- REACT_APP_OPENAI_API_KEY
- REACT_APP_HF_API_TOKEN

If not set at build time, you may paste a key/token into the API Key/Token field on the page.

## Local library import

For development, we import from the local workspace:

- import ... from "../../ai-rag-lite/src/index.ts"

TODO: Switch to a published package or built `dist/` import for production:
- Build the library: cd ai-rag-lite && npm run build
- Update imports to use a published `ai-rag-lite` package or the compiled `dist/` as appropriate.
