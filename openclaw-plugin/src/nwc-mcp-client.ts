import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface } from "node:readline";

type JsonRpcRequest = {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, unknown>;
};

type JsonRpcResponse = {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
};

type NwcToolResult = {
  content: Array<{ type: string; text: string }>;
  details?: Record<string, unknown>;
};

type PendingRequest = {
  resolve: (value: JsonRpcResponse) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const INIT_TIMEOUT_MS = 15_000;
const DEFAULT_TOOL_TIMEOUT_MS = 30_000;

export class NwcMcpClient {
  private process: ChildProcess | null = null;
  private rl: Interface | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private buffer = "";
  private ready = false;
  private readyPromise: Promise<void> | null = null;
  private initResolve: (() => void) | null = null;
  private initReject: ((error: Error) => void) | null = null;
  private died = false;

  constructor(
    private pythonCommand: string,
    private storagePath: string,
    private logger?: { info?: (msg: string) => void; warn?: (msg: string) => void },
  ) {}

  isReady(): boolean {
    return this.ready && !this.died;
  }

  async start(): Promise<void> {
    if (this.ready) return;
    if (this.readyPromise) return this.readyPromise;

    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.initResolve = resolve;
      this.initReject = reject;
    });

    const args = this.pythonCommand.split(/\s+/).filter(Boolean);
    const command = args[0];
    const rest = args.slice(1);
    rest.push("--storage", this.storagePath);

    this.logger?.info?.(`nwcortex: spawning ${command} ${rest.join(" ")}`);

    this.process = spawn(command, rest, {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    this.rl = createInterface({ input: this.process.stdout! });

    this.rl.on("line", (line) => {
      this.handleLine(line);
    });

    this.process.on("error", (err) => {
      this.logger?.warn?.(`nwcortex: process error: ${err.message}`);
      this.died = true;
      this.failAllPending(new Error(`nwc-mcp process error: ${err.message}`));
    });

    this.process.on("close", (code) => {
      this.logger?.warn?.(`nwcortex: process exited (code ${code})`);
      this.died = true;
      this.ready = false;
      this.readyPromise = null;
      this.failAllPending(new Error(`nwc-mcp exited (code ${code})`));
    });

    // Send initialize request
    this.sendRaw({
      jsonrpc: "2.0",
      id: this.nextId++,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "openclaw-nwcortex", version: "1.0.0" },
      },
    });

    // Timeout for initialization
    setTimeout(() => {
      if (!this.ready && this.initReject) {
        this.initReject(new Error("nwc-mcp initialization timed out"));
        this.initReject = null;
        this.initResolve = null;
      }
    }, INIT_TIMEOUT_MS);
  }

  private sendRaw(message: JsonRpcRequest): void {
    if (!this.process?.stdin) return;
    const line = JSON.stringify(message) + "\n";
    this.process.stdin.write(line);
  }

  private handleLine(line: string): void {
    try {
      const msg = JSON.parse(line) as JsonRpcResponse;
      if (typeof msg.id !== "number") return;

      // Handle initialize response
      if (this.initResolve) {
        // Send initialized notification
        this.process?.stdin?.write(
          JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n",
        );
        this.ready = true;
        this.initResolve();
        this.initResolve = null;
        this.initReject = null;
        this.logger?.info?.("nwcortex: MCP initialized");
        return;
      }

      const pending = this.pending.get(msg.id);
      if (!pending) return;

      clearTimeout(pending.timer);
      this.pending.delete(msg.id);

      if (msg.error) {
        pending.reject(new Error(`MCP error ${msg.error.code}: ${msg.error.message}`));
      } else {
        pending.resolve(msg);
      }
    } catch {
      // non-JSON line (e.g., log output) — ignore
    }
  }

  private callTool(
    toolName: string,
    args: Record<string, unknown> = {},
    timeoutMs: number = DEFAULT_TOOL_TIMEOUT_MS,
  ): Promise<JsonRpcResponse> {
    if (this.died) {
      return Promise.reject(new Error("nwc-mcp process is dead"));
    }

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`nwc-mcp tool ${toolName} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      this.sendRaw({
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name: toolName, arguments: args },
      });
    });
  }

  private failAllPending(error: Error): void {
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pending.delete(id);
    }
  }

  async recall(
    query: string,
    opts?: { maxResults?: number },
  ): Promise<{ text: string; details?: Record<string, unknown> }> {
    const resp = await this.callTool("recall", {
      query,
      ...(opts?.maxResults ? { limit: opts.maxResults } : {}),
    });
    const result = resp.result as NwcToolResult | undefined;
    const text = result?.content?.[0]?.text ?? JSON.stringify(resp.result);
    return { text, details: result?.details as Record<string, unknown> | undefined };
  }

  async remember(
    text: string,
    opts?: { tags?: string[]; importance?: number; emotional?: number },
  ): Promise<{ text: string }> {
    const args: Record<string, unknown> = { text };
    if (opts?.tags) args.tags = opts.tags;
    if (opts?.importance !== undefined) args.importance = opts.importance;
    if (opts?.emotional !== undefined) args.emotional = opts.emotional;
    const resp = await this.callTool("remember", args);
    const result = resp.result as NwcToolResult | undefined;
    return { text: result?.content?.[0]?.text ?? JSON.stringify(resp.result) };
  }

  async forget(query: string): Promise<{ text: string }> {
    const resp = await this.callTool("forget", { query });
    const result = resp.result as NwcToolResult | undefined;
    return { text: result?.content?.[0]?.text ?? JSON.stringify(resp.result) };
  }

  async stats(): Promise<Record<string, unknown>> {
    const resp = await this.callTool("stats");
    const result = resp.result as NwcToolResult | undefined;
    try {
      return JSON.parse(result?.content?.[0]?.text ?? "{}") as Record<string, unknown>;
    } catch {
      return { raw: result?.content?.[0]?.text ?? "unknown" };
    }
  }

  async consolidate(): Promise<void> {
    await this.callTool("consolidate", {}, 10_000);
  }

  async sleep(): Promise<string> {
    const resp = await this.callTool("sleep", {}, 60_000);
    const result = resp.result as NwcToolResult | undefined;
    return result?.content?.[0]?.text ?? "";
  }

  async save(): Promise<void> {
    await this.callTool("save", {}, 10_000);
  }

  async load(storagePath?: string): Promise<void> {
    const args: Record<string, unknown> = {};
    if (storagePath) args.path = storagePath;
    await this.callTool("load", args, 10_000);
  }

  async fuzzyRecall(query: string): Promise<{ text: string }> {
    const resp = await this.callTool("fuzzy_recall", { query });
    const result = resp.result as NwcToolResult | undefined;
    return { text: result?.content?.[0]?.text ?? JSON.stringify(resp.result) };
  }

  async getProfile(): Promise<Record<string, unknown>> {
    const resp = await this.callTool("get_profile");
    const result = resp.result as NwcToolResult | undefined;
    try {
      return JSON.parse(result?.content?.[0]?.text ?? "{}") as Record<string, unknown>;
    } catch {
      return { raw: result?.content?.[0]?.text ?? "unknown" };
    }
  }

  rememberWorking(text: string, tags?: string[]): Promise<{ text: string }> {
    return this.callTool("remember_working", { text, ...(tags ? { tags } : {}) }).then((resp) => {
      const result = resp.result as NwcToolResult | undefined;
      return { text: result?.content?.[0]?.text ?? "" };
    });
  }

  async close(): Promise<void> {
    try {
      await this.save();
    } catch {
      // save failed — ignore
    }
    this.failAllPending(new Error("nwc-mcp client closing"));
    this.pending.clear();
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }
    this.ready = false;
    this.readyPromise = null;
    this.died = true;
  }
}
