import { describe, it, expect } from "vitest";
import { LiteRAG } from "../src/rag";
import { MemoryVectorStore } from "../src/store/memoryStore";
import { CustomEmbeddings } from "../src/embeddings/custom";
import { CustomLLM } from "../src/llm/customLLM";

const toyEmbedder = (texts: string[]) => {
  const dim = 32;
  const toVec = (t: string) => {
    const v = new Array(dim).fill(0);
    for (let i = 0; i < t.length; i++) v[t.charCodeAt(i) % dim] += 1;
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / norm);
  };
  return Promise.resolve(texts.map(toVec));
};

describe("LiteRAG", () => {
  it("ingests and answers with context", async () => {
    const rag = new LiteRAG({
      embedder: new CustomEmbeddings(toyEmbedder),
      store: new MemoryVectorStore(),
      llm: new CustomLLM(async (prompt) => {
        expect(prompt).toContain("Context:");
        return "ok";
      }),
    });

    await rag.ingest({
      id: "doc1",
      text: "Paris is the capital of France. The Eiffel Tower is in Paris.",
    });

    const ans = await rag.query("Where is the Eiffel Tower?");
    expect(typeof ans).toBe("string");
  });
});
