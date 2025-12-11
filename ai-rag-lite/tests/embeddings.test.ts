import { describe, it, expect } from "vitest";
import { CustomEmbeddings } from "../src/embeddings/custom";
import { cosineSimilarity } from "../src/search/cosine";
import { OpenAIEmbeddings } from "../src/embeddings/openai";
import { HFEmbeddings } from "../src/embeddings/huggingface";

describe("cosine similarity", () => {
  it("returns 1 for identical vectors", () => {
    const a = [1, 2, 3];
    const b = [1, 2, 3];
    expect(cosineSimilarity(a, b)).toBeCloseTo(1, 6);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = [1, 0];
    const b = [0, 1];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 6);
  });
});

describe("CustomEmbeddings", () => {
  it("is deterministic", async () => {
    const fn = async (texts: string[]) => texts.map((t) => [t.length, t.length * 2]);
    const emb = new CustomEmbeddings(fn);
    const [v1] = await emb.embed(["abc"]);
    const [v2] = await emb.embed(["abc"]);
    expect(v1).toEqual(v2);
  });
});

describe("Provider constructors error when API key/token missing", () => {
  it("OpenAIEmbeddings throws when no key", async () => {
    const old = process.env.OPENAI_API_KEY;
    delete (process.env as any).OPENAI_API_KEY;
    expect(() => new OpenAIEmbeddings()).toThrow();
    if (old) process.env.OPENAI_API_KEY = old;
  });

  it("HFEmbeddings throws when no token", async () => {
    const old = process.env.HUGGINGFACE_API_TOKEN;
    delete (process.env as any).HUGGINGFACE_API_TOKEN;
    expect(() => new HFEmbeddings()).toThrow();
    if (old) process.env.HUGGINGFACE_API_TOKEN = old;
  });
});
