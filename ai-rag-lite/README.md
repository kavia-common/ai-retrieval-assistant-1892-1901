# ai-rag-lite

A lightweight, dependency-minimal Retrieval Augmented Generation (RAG) toolkit that works in both Node and modern browsers. Provides simple components for chunking, embeddings, vector storage, search, and LLM prompting.

- No heavy SDKs by default
- Works with custom embeddings/LLM functions
- Minimal footprint; TypeScript first

## Installation

Clone or copy the `ai-rag-lite` folder into your repo, then:

```bash
cd ai-rag-lite
npm install
npm run build
```

No runtime dependencies are required.

## Environment Variables

For cloud providers (only needed if you use these providers):

- OPENAI_API_KEY: for OpenAI embeddings and chat
- HUGGINGFACE_API_TOKEN: for Hugging Face embeddings

Tests and examples do not require these keys (they use Custom providers).

## Quick Start (Node)

```ts
import {
  LiteRAG,
  MemoryVectorStore,
  CustomEmbeddings,
  CustomLLM
} from "ai-rag-lite";

const embedder = new CustomEmbeddings(async (texts) => {
  const dim = 64;
  const toVec = (t: string) => {
    const v = new Array(dim).fill(0);
    for (let i = 0; i < t.length; i++) v[t.charCodeAt(i) % dim] += 1;
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / n);
  };
  return texts.map(toVec);
});

const llm = new CustomLLM(async (prompt) => "Answer: " + prompt.slice(0, 60) + "...");
const store = new MemoryVectorStore();

const rag = new LiteRAG({ embedder, store, llm });

await rag.ingest({ id: "doc1", text: "The Eiffel Tower is in Paris." });
const answer = await rag.query("Where is the Eiffel Tower?");
console.log(answer);
```

## Quick Start (Browser)

1. Build the library: `npm run build`
2. Serve `dist/` with a dev server that supports ES modules.
3. See `examples/browser-example.html` for a working demo using `CustomEmbeddings` and `MemoryVectorStore`.

## API Overview

- Types
  - Document { id, text, metadata? }
  - Chunk { id, docId, text, embedding?, metadata? }
  - EmbeddingsProvider: `embed(texts: string[]): Promise<number[][]>`
  - VectorStore: `add(chunks)`, `search(queryEmbedding, topK)`
  - LLM: `generate(prompt: string): Promise<string>`

- Chunking
  - `chunkText(doc, { chunkSize=500, chunkOverlap=50 })`

- Search
  - `cosineSimilarity(a, b)`
  - `SimpleHNSWIndex` (approximate; falls back to cosine for small N)

- Vector Stores
  - `MemoryVectorStore` (in-memory)
  - `FileVectorStore` (Node-only, uses fs/promises)

- Embeddings Providers
  - `OpenAIEmbeddings` (fetch)
  - `HFEmbeddings` (fetch, default model sentence-transformers/all-MiniLM-L6-v2)
  - `CustomEmbeddings(fn)`

- LLM Providers
  - `OpenAILLM` (Chat Completions)
  - `CustomLLM(fn)`

- RAG
  - `LiteRAG({ embedder, store, llm })`
    - `ingest(document)`
    - `ingestMany(documents)`
    - `query(question, { topK=5, promptTemplate? })`

## Examples

- Node demo: `examples/simple.js`
- Custom embeddings demo: `examples/custom-embeddings.js`
- Browser ES module demo: `examples/browser-example.html`

## Limitations

- The `SimpleHNSWIndex` is a very lightweight approximation mainly for educational purposes. For large-scale or production-grade ANN, integrate a dedicated ANN library.
- `FileVectorStore` is Node-only and not suitable for browsers.
- Network calls are not mocked in the helpers; tests use only custom providers to avoid network.

## Development

- Build: `npm run build`
- Test: `npm test`
- Dev test watch: `npm run dev`

## License

MIT © 2025
