//
// Cosine similarity utilities and ranking helper
//

/**
 * Compute cosine similarity between two vectors.
 * Returns a number between -1 and 1. If any vector is zero-norm, returns 0.
 */
// PUBLIC_INTERFACE
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] || 0;
    const bi = b[i] || 0;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (!isFinite(denom) || denom === 0) return 0;
  return dot / denom;
}

/**
 * Rank items by cosine similarity.
 * embeddings: array of vectors
 * queryEmbedding: vector
 * returns topK indices with scores
 */
// PUBLIC_INTERFACE
export function rankByCosine(
  embeddings: number[][],
  queryEmbedding: number[],
  topK: number
): Array<{ index: number; score: number }> {
  const scored = embeddings.map((emb, idx) => ({
    index: idx,
    score: cosineSimilarity(emb, queryEmbedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, topK));
}
