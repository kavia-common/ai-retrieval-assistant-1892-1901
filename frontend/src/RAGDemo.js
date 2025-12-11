import React, { useMemo, useRef, useState } from "react";

/**
 * Local workspace import to ai-rag-lite.
 * For development, we import directly from the library source. In production, this should
 * be replaced by a published package or built artifacts.
 *
 * TODO: Replace with a published npm package or prebuilt dist import when available.
 */
import {
  LiteRAG,
  MemoryVectorStore,
  CustomEmbeddings,
  CustomLLM,
  OpenAIEmbeddings,
  HFEmbeddings,
  OpenAILLM,
  chunkText
} from "./lib/ai-rag-lite-proxy";

/**
 * Ocean Professional themed RAG demo.
 * - Supports MemoryVectorStore only (browser)
 * - Embeddings providers: Custom (default), OpenAI, HuggingFace
 * - LLM providers: CustomLLM (default echo), OpenAI if API key present
 */

// PUBLIC_INTERFACE
export default function RAGDemo() {
  const [docsText, setDocsText] = useState("");
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [provider, setProvider] = useState("custom"); // 'custom' | 'openai' | 'hf'
  const [apiKey, setApiKey] = useState("");
  const [question, setQuestion] = useState("Where is the Eiffel Tower?");
  const [ingested, setIngested] = useState(false);
  const [loading, setLoading] = useState(false);
  const [topMatches, setTopMatches] = useState([]);
  const [answer, setAnswer] = useState("");

  // Read optional envs (Create React App exposes REACT_APP_* at build time)
  const envOpenAIKey = process.env.REACT_APP_OPENAI_API_KEY || "";
  const envHFToken = process.env.REACT_APP_HF_API_TOKEN || "";

  // Keep a single store and rag instance by ref
  const storeRef = useRef(null);
  const ragRef = useRef(null);
  const embedderRef = useRef(null);
  const llmRef = useRef(null);

  // Default custom embedding function (simple character-gram)
  const customEmbeddingFn = useMemo(() => {
    return async (texts) => {
      const dim = 64;
      const toVec = (t) => {
        const v = new Array(dim).fill(0);
        for (let i = 0; i < t.length; i++) {
          v[t.charCodeAt(i) % dim] += 1;
        }
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
        return v.map((x) => x / norm);
      };
      return texts.map(toVec);
    };
  }, []);

  // Default custom LLM: echo template-based "answer"
  const customLLMFn = useMemo(() => {
    return async (prompt) => {
      // A very simple "LLM" that echoes the prompt first 160 chars.
      return "CustomLLM (demo): " + String(prompt || "").slice(0, 160) + "...";
    };
  }, []);

  const ensureComponents = () => {
    // Always new store for this demo in browser
    if (!storeRef.current) {
      storeRef.current = new MemoryVectorStore();
    }

    // Embeddings provider selection
    if (!embedderRef.current) {
      if (provider === "openai") {
        const key = apiKey || envOpenAIKey;
        if (!key) {
          // fallback to custom if missing key
          embedderRef.current = new CustomEmbeddings(customEmbeddingFn);
        } else {
          embedderRef.current = new OpenAIEmbeddings({ apiKey: key });
        }
      } else if (provider === "hf") {
        const token = apiKey || envHFToken;
        if (!token) {
          // fallback to custom if missing token
          embedderRef.current = new CustomEmbeddings(customEmbeddingFn);
        } else {
          embedderRef.current = new HFEmbeddings({ apiToken: token });
        }
      } else {
        embedderRef.current = new CustomEmbeddings(customEmbeddingFn);
      }
    }

    // LLM selection: default CustomLLM, OpenAILLM if key available
    if (!llmRef.current) {
      const key = apiKey || envOpenAIKey;
      if (provider === "openai" && key) {
        llmRef.current = new OpenAILLM({ apiKey: key });
      } else {
        llmRef.current = new CustomLLM(customLLMFn);
      }
    }

    if (!ragRef.current) {
      ragRef.current = new LiteRAG({
        embedder: embedderRef.current,
        store: storeRef.current,
        llm: llmRef.current,
        chunkerOptions: {
          chunkSize: Number(chunkSize) || 500,
          chunkOverlap: Number(chunkOverlap) || 50
        }
      });
    }
  };

  const resetPipeline = () => {
    // Reset references when provider/keys/options change for clarity
    storeRef.current = null;
    ragRef.current = null;
    embedderRef.current = null;
    llmRef.current = null;
    setIngested(false);
    setTopMatches([]);
    setAnswer("");
  };

  const onProviderChange = (val) => {
    setProvider(val);
    resetPipeline();
  };

  const onChangeKey = (val) => {
    setApiKey(val);
    resetPipeline();
  };

  const onChangeChunkSize = (val) => {
    setChunkSize(val);
    resetPipeline();
  };

  const onChangeChunkOverlap = (val) => {
    setChunkOverlap(val);
    resetPipeline();
  };

  const ingest = async () => {
    setLoading(true);
    setTopMatches([]);
    setAnswer("");
    try {
      ensureComponents();
      const rag = ragRef.current;
      const text = (docsText || "").trim();
      if (!text) {
        setIngested(false);
        setLoading(false);
        return;
      }
      // We'll break input by two newlines as multiple docs, else single doc
      const parts = text.split(/\n{2,}/).map((t) => t.trim()).filter(Boolean);
      const many = parts.length > 1;
      if (many) {
        const documents = parts.map((t, idx) => ({ id: `doc-${idx + 1}`, text: t }));
        await rag.ingestMany(documents);
      } else {
        await rag.ingest({ id: "doc-1", text });
      }
      setIngested(true);
    } catch (e) {
      console.error(e);
      alert("Ingest failed: " + (e?.message || String(e)));
      setIngested(false);
    } finally {
      setLoading(false);
    }
  };

  const ask = async () => {
    setLoading(true);
    setTopMatches([]);
    setAnswer("");
    try {
      ensureComponents();
      const rag = ragRef.current;

      // We want to also show the top matches; hack: embed the question and search directly via store
      // To get the internal topK matches, we re-run the procedure outside of rag.query.
      const topK = 5;
      const [qEmb] = await embedderRef.current.embed([question]);
      const results = await storeRef.current.search(qEmb, topK);
      setTopMatches(results);

      const res = await rag.query(question, {
        topK,
        // optional custom prompt template to keep it short
        promptTemplate: (q, contexts) => {
          const ctx = contexts.map((c, i) => `(${i + 1}) ${c}`).join("\n\n");
          return [
            "You are a helpful assistant. Answer using only the provided context.",
            "If unsure, say you don't know.",
            "",
            "Context:",
            ctx || "(none)",
            "",
            `Question: ${q}`,
            "",
            "Answer:"
          ].join("\n");
        }
      });
      setAnswer(res);
    } catch (e) {
      console.error(e);
      alert("Query failed: " + (e?.message || String(e)));
    } finally {
      setLoading(false);
    }
  };

  const providerHelp =
    provider === "openai"
      ? "Optional: Provide an OpenAI API Key to use OpenAI embeddings/LLM; otherwise falls back to Custom."
      : provider === "hf"
      ? "Optional: Provide a HuggingFace API Token to use HF embeddings; otherwise falls back to Custom."
      : "Using Custom embeddings/LLM locally (no network calls).";

  return (
    <div className="page">
      <div className="card" style={{ marginBottom: 16 }}>
        <h2 className="title">RAG Demo</h2>
        <p className="subtitle">Ingest text, select provider, and ask a question.</p>
        <div className="form-grid">
          <div className="control">
            <label className="label">Documents (separate multiple docs with blank lines)</label>
            <textarea
              className="textarea"
              rows={8}
              value={docsText}
              onChange={(e) => setDocsText(e.target.value)}
              placeholder="Paste your documents here..."
            />
            <span className="helper">
              Chunking is applied at ingest. Adjust chunk size and overlap below.
            </span>
          </div>

          <div className="control" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div className="control">
              <label className="label">Chunk Size</label>
              <input
                className="input"
                type="number"
                min={50}
                step={10}
                value={chunkSize}
                onChange={(e) => onChangeChunkSize(Number(e.target.value))}
              />
            </div>
            <div className="control">
              <label className="label">Chunk Overlap</label>
              <input
                className="input"
                type="number"
                min={0}
                step={10}
                value={chunkOverlap}
                onChange={(e) => onChangeChunkOverlap(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="control" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div className="control">
              <label className="label">Embeddings/LLM Provider</label>
              <select
                className="select"
                value={provider}
                onChange={(e) => onProviderChange(e.target.value)}
              >
                <option value="custom">Custom (Local)</option>
                <option value="openai">OpenAI</option>
                <option value="hf">HuggingFace (Embeddings only)</option>
              </select>
              <span className="helper">{providerHelp}</span>
            </div>
            <div className="control">
              <label className="label">API Key/Token (optional)</label>
              <input
                className="input"
                type="password"
                value={apiKey}
                onChange={(e) => onChangeKey(e.target.value)}
                placeholder={provider === "openai" ? "OpenAI API Key" : provider === "hf" ? "HF API Token" : "Not required"}
              />
              <span className="helper">
                Environment variables supported: REACT_APP_OPENAI_API_KEY, REACT_APP_HF_API_TOKEN
              </span>
            </div>
          </div>

          <div className="actions">
            <button className="btn primary" onClick={ingest} disabled={loading}>
              {loading ? "Ingesting..." : ingested ? "Re-Ingest" : "Ingest"}
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="form-grid">
          <div className="control">
            <label className="label">Question</label>
            <input
              className="input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question based on the ingested documents"
            />
          </div>
          <div className="actions">
            <button className="btn secondary" onClick={ask} disabled={loading || !ingested}>
              {loading ? "Asking..." : "Ask"}
            </button>
          </div>
        </div>

        <div className="results">
          <div className="panel">
            <h3>Top Matches</h3>
            {topMatches.length === 0 ? (
              <div className="meta">No matches yet.</div>
            ) : (
              topMatches.map((m, idx) => (
                <div className="match" key={idx}>
                  <div className="meta">Score: {m.score.toFixed(4)}</div>
                  <div style={{ whiteSpace: "pre-wrap" }}>{m.chunk?.text}</div>
                </div>
              ))
            )}
          </div>
          <div className="panel">
            <h3>Answer</h3>
            <div style={{ whiteSpace: "pre-wrap" }}>{answer || <span className="meta">No answer yet.</span>}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
