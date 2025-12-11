import { EmbeddingsProvider } from "../types";

/**
 * Hugging Face Inference API embeddings provider using fetch.
 * Default model: sentence-transformers/all-MiniLM-L6-v2
 */

// PUBLIC_INTERFACE
export class HFEmbeddings implements EmbeddingsProvider {
  private apiToken: string;
  private model: string;
  private baseUrl: string;

  constructor(opts?: { apiToken?: string; model?: string }) {
    const token =
      opts?.apiToken ??
      (typeof process !== "undefined" ? process.env.HUGGINGFACE_API_TOKEN : undefined);
    if (!token) {
      throw new Error(
        "HFEmbeddings requires an API token. Set HUGGINGFACE_API_TOKEN or pass apiToken in constructor."
      );
    }
    this.apiToken = token;
    this.model = opts?.model ?? "sentence-transformers/all-MiniLM-L6-v2";
    this.baseUrl = `https://api-inference.huggingface.co/pipeline/feature-extraction/${encodeURIComponent(
      this.model
    )}`;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!Array.isArray(texts) || texts.length === 0) return [];
    const res = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(texts.length === 1 ? texts[0] : texts),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`HuggingFace embeddings failed: ${res.status} ${res.statusText} ${body}`);
    }
    const json = await res.json();
    // API may return [dims] for single input or [[dims], [dims], ...] for batch
    // Normalize to batch shape number[][]
    if (Array.isArray(json) && json.length > 0 && Array.isArray(json[0])) {
      // Determine if it's 2D numeric arrays or 1D for single
      if (typeof json[0][0] === "number") {
        if (typeof json[0][0] === "number" && typeof json[0] === "object" && typeof (json as any)[0][0] === "number") {
          // Could be [dims] single embedding
          // But since json[0][0] is a number and json[0] is array, check if outer is also numbers (single)
        }
      }
    }
    // Normalize robustly:
    const normalize = (out: any): number[][] => {
      if (!Array.isArray(out)) return [];
      if (out.length > 0 && typeof out[0] === "number") {
        return [out as number[]];
      }
      if (out.length > 0 && Array.isArray(out[0])) {
        return out as number[][];
      }
      return [];
    };
    return normalize(json);
  }
}
