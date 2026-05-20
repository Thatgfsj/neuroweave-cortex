// Inline type definitions matching the memory-host-sdk contracts.
// Using structural types so the plugin works without importing internal SDK paths.

type NwcClient = {
  isReady(): boolean;
  recall(query: string, opts?: { maxResults?: number }): Promise<{ text: string; details?: Record<string, unknown> }>;
  stats(): Promise<Record<string, unknown>>;
  consolidate(): Promise<void>;
  close(): Promise<void>;
};

type MemorySource = "memory" | "sessions";

type MemorySearchResult = {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  vectorScore?: number;
  textScore?: number;
  snippet: string;
  source: MemorySource;
  citation?: string;
};

type MemoryReadResult = {
  text: string;
  path: string;
  truncated?: boolean;
  from?: number;
  lines?: number;
  nextFrom?: number;
};

type MemoryProviderStatus = {
  backend: "builtin" | "qmd";
  provider: string;
  model?: string;
  files?: number;
  chunks?: number;
  workspaceDir?: string;
  dbPath?: string;
  custom?: Record<string, unknown>;
};

type MemoryEmbeddingProbeResult = {
  ok: boolean;
  error?: string;
  checked?: boolean;
  cached?: boolean;
  checkedAtMs?: number;
  cacheExpiresAtMs?: number;
};

// ============================================================================

export class NwcMemoryManager {
  private statusCache: { data: MemoryProviderStatus; cachedAt: number } | null = null;
  private statusTtlMs = 30_000;

  constructor(private client: NwcClient) {}

  async search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      sources?: Array<"memory" | "sessions">;
    },
  ): Promise<MemorySearchResult[]> {
    if (!this.client.isReady()) return [];

    const maxResults = opts?.maxResults ?? 5;
    const minScore = opts?.minScore ?? 0;

    try {
      const result = await this.client.recall(query, { maxResults });
      return this.parseRecallText(result.text, minScore);
    } catch {
      return [];
    }
  }

  private parseRecallText(text: string, minScore: number): MemorySearchResult[] {
    const entries = text.split("\n").filter((line) => /^\d+\./.test(line.trim()));
    const results: MemorySearchResult[] = [];

    for (const entry of entries) {
      const match = entry.match(/^\d+\.\s*(\[.*?\])?\s*(.+?)(?:\s*\((\d+)%\))?$/);
      if (!match) continue;

      const snippet = (match[2] ?? entry).trim();
      const scoreStr = match[3];
      const score = scoreStr ? Number(scoreStr) / 100 : 0.5;
      if (score < minScore) continue;

      results.push({
        path: `nwc:recall:${results.length}`,
        startLine: 1,
        endLine: 1,
        score,
        snippet,
        source: "memory",
      });
    }

    if (results.length === 0 && text.trim()) {
      results.push({
        path: "nwc:recall:0",
        startLine: 1,
        endLine: 1,
        score: 0.5,
        snippet: text.trim(),
        source: "memory",
      });
    }

    return results;
  }

  async readFile(params: { relPath: string; from?: number; lines?: number }): Promise<MemoryReadResult> {
    try {
      const result = await this.client.recall(params.relPath, { maxResults: 1 });
      return { text: result.text, path: params.relPath };
    } catch {
      return { text: "", path: params.relPath };
    }
  }

  status(): MemoryProviderStatus {
    return (
      this.statusCache?.data ?? { backend: "qmd", provider: "nwcortex" }
    );
  }

  async sync(_params?: { reason?: string }): Promise<void> {
    // Refresh status cache from live stats before consolidation
    try {
      const stats = await this.client.stats();
      this.statusCache = {
        data: { backend: "qmd", provider: "nwcortex", custom: { nwcStats: stats } },
        cachedAt: Date.now(),
      };
    } catch {
      // keep stale cache on failure
    }
    await this.client.consolidate();
  }

  getCachedEmbeddingAvailability(): MemoryEmbeddingProbeResult | null {
    if (!this.client.isReady()) {
      return { ok: false, error: "nwc-mcp not connected" };
    }
    return { ok: true, cached: true, checkedAtMs: Date.now() };
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    const ready = this.client.isReady();
    return {
      ok: ready,
      error: ready ? undefined : "nwc-mcp not connected",
      checked: true,
      checkedAtMs: Date.now(),
    };
  }

  async probeVectorAvailability(): Promise<boolean> {
    return this.client.isReady();
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}
