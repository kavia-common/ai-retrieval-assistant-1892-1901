import { VectorStore, Chunk } from "../types";
import { rankByCosine } from "../search/cosine";

/**
 * Simple in-memory vector store.
 * Stores chunks and their embeddings side by side.
 */

// PUBLIC_INTERFACE
export class MemoryVectorStore implements VectorStore {
  private chunks: Chunk[] = [];
  private embeddings: number[][] = [];

  async add(chunks: Chunk[]): Promise<void> {
    for (const c of chunks) {
      if (!c.embedding || !Array.isArray(c.embedding)) {
        throw new Error("MemoryVectorStore.add requires chunks to have embedding vectors.");
      }
      this.chunks.push(c);
      this.embeddings.push(c.embedding);
    }
  }

  async search(
    queryEmbedding: number[],
    topK: number
  ): Promise<Array<{ chunk: Chunk; score: number }>> {
    const ranked = rankByCosine(this.embeddings, queryEmbedding, topK);
    return ranked.map(({ index, score }) => ({ chunk: this.chunks[index], score }));
  }

  /** helper for tests/introspection */
  // PUBLIC_INTERFACE
  size(): number {
    return this.chunks.length;
  }
}
