import { LLM } from "../types";

/**
 * Custom LLM wrapper for user-provided generation function.
 */

// PUBLIC_INTERFACE
export class CustomLLM implements LLM {
  private fn: (prompt: string) => Promise<string>;

  constructor(fn: (prompt: string) => Promise<string>) {
    this.fn = fn;
  }

  async generate(prompt: string): Promise<string> {
    return this.fn(prompt);
  }
}
