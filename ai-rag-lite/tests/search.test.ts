import { describe, it, expect } from "vitest";
import { MemoryVectorStore } from "../src/store/memoryStore";
import { Chunk } from "../src/types";

describe("MemoryVectorStore", () => {
  it("add and search returns correct ranking", async () => {
    const store = new MemoryVectorStore();
    const chunks: Chunk[] = [
      { id: "c1", docId: "d", text: "a", embedding: [1, 0] },
      { id: "c2", docId: "d", text: "b", embedding: [0, 1] },
      { id: "c3", docId: "d", text: "c", embedding: [0.7, 0.7] },
    ];
    await store.add(chunks);

    const res = await store.search([1, 0], 2);
    expect(res.length).toBe(2);
    expect(res[0].chunk.id).toBe("c1"); // most similar to [1,0]
  });
});
