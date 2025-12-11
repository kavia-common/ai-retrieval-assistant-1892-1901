//
// ai-rag-lite: Shared types and public interfaces
//

// PUBLIC_INTERFACE
export interface Document {
  /** Unique identifier for the document */
  id: string;
  /** Full text content of the document */
  text: string;
  /** Optional metadata associated with the document */
  metadata?: Record<string, any>;
}

// PUBLIC_INTERFACE
export interface Chunk {
  /** Unique identifier for the chunk */
  id: string;
  /** Reference to source document */
  docId: string;
  /** The chunked text content */
  text: string;
  /** Optional precomputed embedding vector */
  embedding?: number[];
  /** Optional metadata */
  metadata?: Record<string, any>;
}

// PUBLIC_INTERFACE
export interface EmbeddingsProvider {
  /**
   * Generate embeddings for a list of texts.
   * Must be deterministic for the same text/model/tokenization.
   */
  embed(texts: string[]): Promise<number[][]>;
}

// PUBLIC_INTERFACE
export interface VectorStore {
  /**
   * Add chunks to the vector store.
   * If embeddings are not present on chunks, the caller is responsible for computing and providing them.
   */
  add(chunks: Chunk[]): Promise<void>;

  /**
   * Search the vector store using a query embedding.
   * Returns topK results as an array of {chunk, score}, where higher score means more similar.
   */
  search(queryEmbedding: number[], topK: number): Promise<Array<{ chunk: Chunk; score: number }>>;
}

// PUBLIC_INTERFACE
export interface SearchIndex {
  /**
   * Build or update the index with provided embeddings.
   */
  build(embeddings: number[][]): void;

  /**
   * Query by vector, returning indices and scores.
   */
  query(vector: number[], topK: number): Array<{ index: number; score: number }>;
}

// PUBLIC_INTERFACE
export interface LLM {
  /**
   * Generate a text completion based on a given prompt.
   */
  generate(prompt: string): Promise<string>;
}

// PUBLIC_INTERFACE
export interface ChunkerOptions {
  /** Desired character size of each chunk (default 500) */
  chunkSize?: number;
  /** Desired character overlap between chunks (default 50) */
  chunkOverlap?: number;
  /** Optional sentence boundary regex (default /[.!?]\s+/) */
  sentenceBoundaryRegex?: RegExp;
}
