import { LiteRAG, MemoryVectorStore, CustomEmbeddings, CustomLLM /*, OpenAIEmbeddings, OpenAILLM*/ } from "../dist/index.js";

// Simple character-gram embedding for demo (deterministic)
function toyEmbedder(texts) {
  const dim = 64;
  const toVec = (t) => {
    const v = new Array(dim).fill(0);
    for (let i = 0; i < t.length; i++) {
      const code = t.charCodeAt(i);
      v[code % dim] += 1;
    }
    // L2 normalize
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / norm);
  };
  return Promise.resolve(texts.map(toVec));
}

async function main() {
  // Optional: switch to OpenAI by uncommenting and setting env OPENAI_API_KEY
  // const embedder = new OpenAIEmbeddings();
  // const llm = new OpenAILLM();

  const embedder = new CustomEmbeddings(toyEmbedder);
  const llm = new CustomLLM(async (prompt) => {
    // naive "LLM": just returns a canned response with first 100 chars of prompt hash
    return "Demo answer (no external API call). Prompt length: " + prompt.length;
  });

  const store = new MemoryVectorStore();
  const rag = new LiteRAG({ embedder, store, llm });

  await rag.ingest({
    id: "doc1",
    text: "The Eiffel Tower is in Paris. It is a wrought-iron lattice tower on the Champ de Mars.",
  });

  const answer = await rag.query("Where is the Eiffel Tower?");
  console.log("Answer:", answer);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
