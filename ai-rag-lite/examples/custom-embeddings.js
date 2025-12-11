import { LiteRAG, MemoryVectorStore, CustomEmbeddings, CustomLLM } from "../dist/index.js";

// Simple bag-of-words over fixed vocab
const vocab = ["cat", "dog", "paris", "tower", "coffee", "pizza", "eiffel"];
const dim = vocab.length;
const vocabIndex = new Map(vocab.map((w, i) => [w, i]));

function bowEmbedder(texts) {
  const toVec = (t) => {
    const v = new Array(dim).fill(0);
    const tokens = t.toLowerCase().split(/[^a-z]+/).filter(Boolean);
    for (const tok of tokens) {
      const i = vocabIndex.get(tok);
      if (i !== undefined) v[i] += 1;
    }
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / norm);
  };
  return Promise.resolve(texts.map(toVec));
}

async function run() {
  const rag = new LiteRAG({
    embedder: new CustomEmbeddings(bowEmbedder),
    store: new MemoryVectorStore(),
    llm: new CustomLLM(async (prompt) => "Pretend LLM says: " + prompt.slice(0, 80) + "..."),
  });

  await rag.ingest({
    id: "d1",
    text: "The Eiffel Tower is in Paris and it is a famous landmark.",
  });
  await rag.ingest({
    id: "d2",
    text: "Cats and dogs are common pets.",
  });

  const res = await rag.query("Where is the Eiffel Tower located?", { topK: 3 });
  console.log(res);
}

run();
