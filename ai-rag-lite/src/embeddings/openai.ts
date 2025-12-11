import { EmbeddingsProvider } from "../types";

/**
 * Minimal OpenAI embeddings provider using fetch.
 * No OpenAI SDK dependency.
 */

// PUBLIC_INTERFACE
export class OpenAIEmbeddings implements EmbeddingsProvider {
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(opts?: { apiKey?: string; model?: string; baseUrl?: string }) {
    const key = opts?.apiKey ?? (typeof process !== "undefined" ? process.env.OPENAI_API_KEY : undefined);
    if (!key) {
      throw new Error(
        "OpenAIEmbeddings requires an API key. Set OPENAI_API_KEY environment variable or pass apiKey in constructor."
      );
    }
    this.apiKey = key;
    this.model = opts?.model ?? "text-embedding-3-small";
    this.baseUrl = opts?.baseUrl ?? "https://api.openai.com/v1/embeddings";
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!Array.isArray(texts) || texts.length === 0) return [];
    const res = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
      }),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`OpenAI embeddings failed: ${res.status} ${res.statusText} ${body}`);
    }

    const json = (await res.json()) as any;
    if (!json || !Array.isArray(json.data)) {
      throw new Error("Unexpected response from OpenAI embeddings API");
    }
    return json.data.map((d: any) => d.embedding as number[]);
  }
}
