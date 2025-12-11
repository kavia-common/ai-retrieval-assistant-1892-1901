import { ChunkerOptions, Document, Chunk } from "./types";

/**
 * Simple chunker that splits text by sentence boundaries with approximate character limits.
 * Default chunkSize=500 and chunkOverlap=50.
 */

// PUBLIC_INTERFACE
export function chunkText(
  doc: Document,
  options: ChunkerOptions = {}
): Chunk[] {
  const {
    chunkSize = 500,
    chunkOverlap = 50,
    sentenceBoundaryRegex = /(?<=[.!?])\s+/
  } = options;

  const sentences = doc.text
    .split(sentenceBoundaryRegex)
    .map((s) => s.trim())
    .filter(Boolean);

  const chunks: Chunk[] = [];
  let current = "";
  let chunkIndex = 0;

  const pushChunk = (text: string) => {
    const id = `${doc.id}::${chunkIndex++}`;
    chunks.push({
      id,
      docId: doc.id,
      text,
      metadata: doc.metadata ? { ...doc.metadata } : undefined
    });
  };

  for (const sentence of sentences) {
    if ((current + " " + sentence).trim().length <= chunkSize) {
      current = (current ? current + " " : "") + sentence;
    } else {
      if (current) {
        pushChunk(current);
        // overlap: take last chunkOverlap characters from current
        const overlapText =
          chunkOverlap > 0
            ? current.slice(Math.max(0, current.length - chunkOverlap))
            : "";
        current = (overlapText + " " + sentence).trim();
      } else {
        // sentence longer than chunk size; slice
        let start = 0;
        while (start < sentence.length) {
          const slice = sentence.slice(start, start + chunkSize);
          pushChunk(slice);
          const overlap =
            chunkOverlap > 0
              ? slice.slice(Math.max(0, slice.length - chunkOverlap))
              : "";
          start += chunkSize - overlap.length || chunkSize;
        }
        current = "";
      }
    }
  }

  if (current) pushChunk(current);

  return chunks;
}
