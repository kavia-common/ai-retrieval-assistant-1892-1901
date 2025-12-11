import { EmbeddingsProvider } from "../types";

/**
 * Custom embeddings provider wrapper.
 * Accepts a function that maps texts[] -> Promise<number[][]>
 */

// PUBLIC_INTERFACE
export class CustomEmbeddings implements EmbeddingsProvider {
  private fn: (texts: string[]) => Promise<number[][]>;

  constructor(fn: (texts: string[]) => Promise<number[][]>) {
    this.fn = fn;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return this.fn(texts);
  }
}
