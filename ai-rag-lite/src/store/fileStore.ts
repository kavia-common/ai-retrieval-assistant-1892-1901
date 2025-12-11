import { VectorStore, Chunk } from "../types";
import { rankByCosine } from "../search/cosine";

/**
 * Node-only JSON file-based vector store.
 * Persists chunks and embeddings to a file path.
 * In the browser environment, throws an informative error upon construction.
 */

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof (window as any).document !== "undefined";
}

// PUBLIC_INTERFACE
export class FileVectorStore implements VectorStore {
  private filePath: string;
  private chunks: Chunk[] = [];
  private embeddings: number[][] = [];
  private loaded = false;

  constructor(filePath: string) {
    if (isBrowser()) {
      throw new Error("FileVectorStore is not available in browser environments.");
    }
    this.filePath = filePath;
  }

  private async load(): Promise<void> {
    if (this.loaded) return;
    // dynamic import to avoid bundlers including fs in browser builds
    const fs = await import("fs/promises");
    try {
      const data = await fs.readFile(this.filePath, "utf-8");
      const json = JSON.parse(data);
      this.chunks = json.chunks || [];
      this.embeddings = json.embeddings || [];
    } catch (err: any) {
      if (err && err.code === "ENOENT") {
        this.chunks = [];
        this.embeddings = [];
      } else {
        throw err;
      }
    } finally {
      this.loaded = true;
    }
  }

  private async save(): Promise<void> {
    const fs = await import("fs/promises");
    const json = JSON.stringify(
      { chunks: this.chunks, embeddings: this.embeddings },
      null,
      2
    );
    await fs.writeFile(this.filePath, json, "utf-8");
  }

  async add(chunks: Chunk[]): Promise<void> {
    await this.load();
    for (const c of chunks) {
      if (!c.embedding || !Array.isArray(c.embedding)) {
        throw new Error("FileVectorStore.add requires chunks to have embedding vectors.");
      }
      this.chunks.push(c);
      this.embeddings.push(c.embedding);
    }
    await this.save();
  }

  async search(
    queryEmbedding: number[],
    topK: number
  ): Promise<Array<{ chunk: Chunk; score: number }>> {
    await this.load();
    const ranked = rankByCosine(this.embeddings, queryEmbedding, topK);
    return ranked.map(({ index, score }) => ({ chunk: this.chunks[index], score }));
  }
}
