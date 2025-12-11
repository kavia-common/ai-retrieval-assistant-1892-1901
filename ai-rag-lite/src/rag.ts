import { Chunk, ChunkerOptions, Document, EmbeddingsProvider, LLM, VectorStore } from "./types";
import { chunkText } from "./chunker";

/**
 * LiteRAG orchestrator:
 * - Ingest documents: chunk -> embed -> store
 * - Query: embed -> retrieve -> prompt -> generate
 */

// PUBLIC_INTERFACE
export class LiteRAG {
  private embedder: EmbeddingsProvider;
  private store: VectorStore;
  private llm: LLM;
  private chunkerOptions?: ChunkerOptions;

  constructor(opts: {
    embedder: EmbeddingsProvider;
    store: VectorStore;
    llm: LLM;
    chunkerOptions?: ChunkerOptions;
  }) {
    this.embedder = opts.embedder;
    this.store = opts.store;
    this.llm = opts.llm;
    this.chunkerOptions = opts.chunkerOptions;
  }

  // PUBLIC_INTERFACE
  async ingest(document: Document): Promise<Chunk[]> {
    const chunks = chunkText(document, this.chunkerOptions);
    const embeddings = await this.embedder.embed(chunks.map((c) => c.text));
    const withEmbeddings: Chunk[] = chunks.map((c, i) => ({ ...c, embedding: embeddings[i] }));
    await this.store.add(withEmbeddings);
    return withEmbeddings;
  }

  // PUBLIC_INTERFACE
  async ingestMany(documents: Document[]): Promise<Chunk[][]> {
    const all: Chunk[][] = [];
    for (const doc of documents) {
      all.push(await this.ingest(doc));
    }
    return all;
  }

  // PUBLIC_INTERFACE
  async query(
    question: string,
    opts?: {
      topK?: number;
      promptTemplate?: (question: string, contexts: string[]) => string;
    }
  ): Promise<string> {
    const topK = opts?.topK ?? 5;
    const promptTemplate =
      opts?.promptTemplate ?? defaultPromptTemplate;

    // embed the question
    const [qEmb] = await this.embedder.embed([question]);
    const results = await this.store.search(qEmb, topK);
    const contexts = results.map((r) => r.chunk.text);

    const prompt = promptTemplate(question, contexts);
    return this.llm.generate(prompt);
  }
}

function defaultPromptTemplate(question: string, contexts: string[]): string {
  const contextBlock = contexts.map((c, i) => `Snippet ${i + 1}:\n${c}`).join("\n\n");
  return [
    "You are a helpful assistant. Answer the question using only the provided context snippets.",
    "If the answer cannot be found in the context, say you don't know.",
    "",
    "Context:",
    contextBlock || "(no context)",
    "",
    `Question: ${question}`,
    "",
    "Answer concisely:",
  ].join("\n");
}
