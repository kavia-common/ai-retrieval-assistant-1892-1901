import { LLM } from "../types";

/**
 * Minimal OpenAI Chat Completions wrapper using fetch.
 * Default model: gpt-4o-mini (cost-effective). Fallback: gpt-3.5-turbo if needed by user configuration.
 */

// PUBLIC_INTERFACE
export class OpenAILLM implements LLM {
  private apiKey: string;
  private model: string;
  private baseUrl: string;

  constructor(opts?: { apiKey?: string; model?: string; baseUrl?: string }) {
    const key = opts?.apiKey ?? (typeof process !== "undefined" ? process.env.OPENAI_API_KEY : undefined);
    if (!key) {
      throw new Error(
        "OpenAILLM requires an API key. Set OPENAI_API_KEY environment variable or pass apiKey in constructor."
      );
    }
    this.apiKey = key;
    this.model = opts?.model ?? "gpt-4o-mini";
    this.baseUrl = opts?.baseUrl ?? "https://api.openai.com/v1/chat/completions";
  }

  async generate(prompt: string): Promise<string> {
    const res = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.2,
      }),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`OpenAI Chat failed: ${res.status} ${res.statusText} ${body}`);
    }
    const json = (await res.json()) as any;
    const text =
      json?.choices?.[0]?.message?.content ??
      json?.choices?.[0]?.delta?.content ??
      "";
    return typeof text === "string" ? text : String(text ?? "");
  }
}
