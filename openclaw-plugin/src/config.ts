export type NwcConfig = {
  storagePath: string;
  pythonCommand: string;
  autoRecall: boolean;
  autoCapture: boolean;
};

export const DEFAULT_PYTHON_COMMAND = "nwc-mcp";
export const DEFAULT_STORAGE_PATH = "~/.openclaw/nwcortex/memory.json";
export const DEFAULT_AUTO_RECALL = false;
export const DEFAULT_AUTO_CAPTURE = false;

export function resolveNwcConfig(pluginConfig: unknown): NwcConfig {
  const cfg = pluginConfig as Record<string, unknown> | null | undefined;
  return {
    storagePath:
      typeof cfg?.storagePath === "string" && cfg.storagePath.length > 0
        ? cfg.storagePath
        : DEFAULT_STORAGE_PATH,
    pythonCommand:
      typeof cfg?.pythonCommand === "string" && cfg.pythonCommand.length > 0
        ? cfg.pythonCommand
        : DEFAULT_PYTHON_COMMAND,
    autoRecall: typeof cfg?.autoRecall === "boolean" ? cfg.autoRecall : DEFAULT_AUTO_RECALL,
    autoCapture: typeof cfg?.autoCapture === "boolean" ? cfg.autoCapture : DEFAULT_AUTO_CAPTURE,
  };
}
