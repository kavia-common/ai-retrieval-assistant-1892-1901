import { rankByCosine } from "./cosine";

/**
 * Lightweight approximate search stub.
 * For small datasets, falls back to exact cosine ranking.
 * For larger datasets, performs a simple multi-start greedy search on a synthetic neighbor graph
 * constructed via random links. Dependency-free and deterministic with fixed seed per build.
 */

type Node = {
  index: number;
  neighbors: number[];
};

function pseudoRandom(seed: number) {
  // simple LCG to keep deterministic
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

// PUBLIC_INTERFACE
export class SimpleHNSWIndex {
  private nodes: Node[] = [];
  private dims = 0;
  private embeddings: number[][] = [];
  private built = false;
  private maxNeighbors: number;

  constructor(maxNeighbors = 8) {
    this.maxNeighbors = Math.max(2, maxNeighbors | 0);
  }

  build(embeddings: number[][]) {
    if (embeddings.length === 0) {
      this.nodes = [];
      this.embeddings = [];
      this.dims = 0;
      this.built = true;
      return;
    }
    this.embeddings = embeddings;
    this.dims = embeddings[0].length;
    const n = embeddings.length;
    const rnd = pseudoRandom(n ^ this.dims ^ 1234567);

    // Build a random neighbor graph
    this.nodes = Array.from({ length: n }, (_, i) => ({
      index: i,
      neighbors: [],
    }));

    for (let i = 0; i < n; i++) {
      const deg = Math.min(this.maxNeighbors, Math.max(2, Math.floor(rnd() * this.maxNeighbors) + 2));
      const neigh = new Set<number>();
      while (neigh.size < deg) {
        const candidate = Math.floor(rnd() * n);
        if (candidate !== i) neigh.add(candidate);
      }
      this.nodes[i].neighbors = Array.from(neigh);
    }
    this.built = true;
  }

  query(vector: number[], topK: number): Array<{ index: number; score: number }> {
    if (!this.built) throw new Error("Index not built");
    const n = this.embeddings.length;
    if (n === 0) return [];
    // For small n, use exact ranking
    if (n <= 64) {
      return rankByCosine(this.embeddings, vector, topK);
    }

    // Multi-start greedy search
    const starts = Math.min(5, Math.max(1, Math.floor(n / 200)));
    const visited = new Set<number>();
    const candidates = new Set<number>();
    const scores: Record<number, number> = {};

    const rand = pseudoRandom(n ^ vector.length ^ 98765);
    const cosine = (a: number[], b: number[]) => {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) {
        const ai = a[i] || 0, bi = b[i] || 0;
        dot += ai * bi; na += ai * ai; nb += bi * bi;
      }
      const denom = Math.sqrt(na) * Math.sqrt(nb);
      if (!isFinite(denom) || denom === 0) return 0;
      return dot / denom;
    };

    for (let s = 0; s < starts; s++) {
      let current = Math.floor(rand() * n);
      let improved = true;
      while (improved) {
        visited.add(current);
        const currentScore =
          scores[current] ?? (scores[current] = cosine(this.embeddings[current], vector));
        let best = { idx: current, score: currentScore };
        for (const nb of this.nodes[current].neighbors) {
          if (visited.has(nb)) continue;
          const sc = scores[nb] ?? (scores[nb] = cosine(this.embeddings[nb], vector));
          if (sc > best.score) {
            best = { idx: nb, score: sc };
          }
          candidates.add(nb);
        }
        improved = best.idx !== current;
        current = best.idx;
      }
      candidates.add(current);
    }

    const scored = Array.from(candidates).map((idx) => ({
      index: idx,
      score: scores[idx] ?? (scores[idx] = cosine(this.embeddings[idx], vector)),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.max(0, topK));
  }
}
