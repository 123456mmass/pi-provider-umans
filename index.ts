import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type {
  ProviderModelConfig,
  OAuthCredentials,
  OAuthLoginCallbacks,
  AssistantMessage,
} from "@earendil-works/pi-ai";

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

const MODELS_INFO_URL = "https://api.code.umans.ai/v1/models/info";
const USAGE_URL = "https://api.code.umans.ai/v1/usage";

// ---------------------------------------------------------------------------
// Vision Bridge — describe images for text-only models via a vision-capable
// model on the 9router gateway before forwarding to the Umans API.
// ---------------------------------------------------------------------------
const VISION_BRIDGE_GATEWAY = "https://gateway.birdsphichitchai.dev/v1";
const VISION_BRIDGE_KEY = "sk-eccd39459d7e83c3-va0s0e-e2effa32";
const VISION_BRIDGE_MODEL = "ag/gemini-3.5-flash-low"; // fast vision-capable, low cost
const VISION_BRIDGE_MAX_TOKENS = 1024;
const VISION_BRIDGE_PROMPT =
  "You are a vision bridge for a coding agent that cannot see images. " +
  "Describe this image in detail. Focus on: visible text (especially code, error messages, labels), " +
  "UI elements and layout, colors, shapes, and any data. " +
  "Be precise and concise — this description replaces the image for the agent. " +
  "Do NOT add commentary about the task, just describe what you see.";

// Models known to support vision natively (skip the bridge for these).
const NATIVE_VISION_MODELS = new Set<string>([
  "umans-coder",
  "umans-kimi-k2.5",
  "umans-kimi-k2.6",
  "umans-kimi-k2.7",
  "umans-flash",
]);

function isTextOnlyUmansModel(modelId: string): boolean {
  return !NATIVE_VISION_MODELS.has(modelId);
}

// Cache image descriptions by a hash of the image data to avoid re-describing.
const imageDescriptionCache = new Map<string, string>();

/** Fast hash of the full image URL — avoids cache collisions between
 *  different PNG screenshots that share the same header prefix. */
function imageHash(str: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(16).padStart(8, "0") + (h1 >>> 0).toString(16).padStart(8, "0");
}

async function describeImageViaGateway(imageUrl: string): Promise<{ description: string; cached: boolean }> {
  // Hash the full URL to avoid cache collisions (PNG screenshots share
  // the same header prefix, so slicing would key different images alike)
  const cacheKey = imageHash(imageUrl);
  const cached = imageDescriptionCache.get(cacheKey);
  if (cached) return { description: cached, cached: true };

  const res = await fetch(`${VISION_BRIDGE_GATEWAY}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${VISION_BRIDGE_KEY}`,
    },
    body: JSON.stringify({
      model: VISION_BRIDGE_MODEL,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: VISION_BRIDGE_PROMPT },
          { type: "image_url", image_url: { url: imageUrl } },
        ],
      }],
      stream: false,
      max_tokens: VISION_BRIDGE_MAX_TOKENS,
      thinking: { type: "disabled" },
    }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`vision bridge HTTP ${res.status}: ${errText.slice(0, 200)}`);
  }
  const data = await res.json();
  const description = data.choices?.[0]?.message?.content || "(image description unavailable)";
  imageDescriptionCache.set(cacheKey, description);
  return { description, cached: false };
}

/** Run async tasks with limited concurrency to avoid overwhelming the gateway. */
async function mapWithConcurrency<T, R>(
  items: T[],
  fn: (item: T, index: number) => Promise<R>,
  concurrency = 3,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker()),
  );
  return results;
}

const FALLBACK_MODELS: ProviderModelConfig[] = [
  {
    id: "umans-coder",
    name: "Umans Coder",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 32768,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  },
  {
    id: "umans-kimi-k2.5",
    name: "Umans Kimi K2.5",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 32768,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  },
  {
    id: "umans-kimi-k2.6",
    name: "Umans Kimi K2.6",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 32768,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  },
  {
    id: "umans-glm-5.1",
    name: "Umans GLM 5.1",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 202752,
    maxTokens: 131072,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  },
  {
    id: "umans-glm-5.2",
    name: "Umans GLM 5.2",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 1000000,
    maxTokens: 32768,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
    thinkingLevelMap: { minimal: null, low: null, medium: null, high: "high", xhigh: "max" },
  },
  {
    id: "umans-minimax-m2.5",
    name: "Umans MiniMax M2.5",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 204800,
    maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  },
];

/**
 * Build reasoning config from the Umans `/models/info` capability payload.
 *
 * Umans advertises per-model `reasoning.levels` (e.g. GLM 5.2: ["none","high","max"]).
 * pi thinking levels map onto those Umans values:
 *   pi "off"    -> Umans "none"   (disable reasoning)
 *   pi "xhigh"  -> Umans "max"     (maximum reasoning)
 *   minimal/low/medium/high -> same name when the model advertises it, else hidden.
 *
 * Models with no advertised levels (e.g. Kimi, `levels: []`) are on/off only:
 * `supportsReasoningEffort` is false so pi sends just `thinking: { type: "enabled"|"disabled" }`.
 *
 * Verified against the live Umans endpoint: GLM accepts reasoning_effort (none/high/max),
 * Kimi ignores unknown values. The previous reasoning_effort strip is therefore removed.
 */
function buildReasoningConfig(reasoning: any): {
  thinkingLevelMap?: ProviderModelConfig["thinkingLevelMap"];
  supportsReasoningEffort: boolean;
} {
  const levels: string[] = Array.isArray(reasoning?.levels) ? reasoning.levels : [];
  if (levels.length === 0) {
    return { supportsReasoningEffort: false };
  }
  const canDisable = reasoning?.can_disable !== false;
  const has = (v: string) => levels.includes(v);
  return {
    thinkingLevelMap: {
      off: has("none") && canDisable ? "none" : null,
      minimal: null, // Umans has no "minimal" tier
      low: has("low") ? "low" : null,
      medium: has("medium") ? "medium" : null,
      high: has("high") ? "high" : null,
      xhigh: has("max") ? "max" : null,
    },
    supportsReasoningEffort: true,
  };
}

function mapUmansModel(id: string, info: any): ProviderModelConfig {
  const caps = info.capabilities ?? {};
  const supportsVision = caps.supports_vision === true;

  // recommended_max_tokens = max output tokens. Fall back to 65000 if missing or < 8192.
  const recommendedMax = caps.recommended_max_tokens;
  const maxTokens: number =
    typeof recommendedMax === "number" && recommendedMax >= 8192
      ? recommendedMax
      : 65000;

  const { thinkingLevelMap, supportsReasoningEffort } = buildReasoningConfig(caps.reasoning);

  const result: ProviderModelConfig = {
    id,
    name: info.display_name || id,
    reasoning: true,
    // Always claim image support so pi keeps images in the payload.
    // The vision bridge in before_provider_request describes images for
    // text-only models before they reach the Umans API.
    input: ["text", "image"],
    contextWindow: caps.context_window ?? 200000,
    maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    thinkingLevelMap,
    compat: { supportsDeveloperRole: false, supportsReasoningEffort, thinkingFormat: "deepseek", requiresReasoningContentOnAssistantMessages: true },
  };

  // Track native vision models so the bridge can skip them
  if (supportsVision) {
    NATIVE_VISION_MODELS.add(id);
  }

  return result;
}

// Dynamic model fetch at module load time
let models: ProviderModelConfig[] = FALLBACK_MODELS;

try {
  const res = await fetch(MODELS_INFO_URL, { signal: AbortSignal.timeout(5000) });
  if (res.ok) {
    const data = await res.json();
    models = Object.entries(data).map(([id, info]) =>
      mapUmansModel(id, info as any),
    );
  } else {
    console.warn(
      `[pi-provider-umans] Models API returned ${res.status}, using fallback`,
    );
  }
} catch (err) {
  console.warn(
    "[pi-provider-umans] Failed to fetch dynamic models, using fallback:",
    err,
  );
}

// ---------------------------------------------------------------------------
// OAuth (API key stored in auth.json)
// ---------------------------------------------------------------------------

async function loginUmans(
  callbacks: OAuthLoginCallbacks,
): Promise<OAuthCredentials> {
  const apiKey = await callbacks.onPrompt({
    message: "Enter your Umans API key (starts with sk-):",
  });
  const key = apiKey.trim();
  if (!key.startsWith("sk-")) {
    throw new Error("Invalid API key: must start with 'sk-'");
  }
  // API keys don't expire — use far-future timestamp to avoid unnecessary refresh attempts
  return { refresh: key, access: key, expires: Date.now() + 100 * 365 * 24 * 60 * 60 * 1000 };
}

function refreshUmansToken(
  credentials: OAuthCredentials,
): Promise<OAuthCredentials> {
  return Promise.resolve(credentials);
}

function getApiKey(credentials: OAuthCredentials): string {
  return credentials.access;
}

// ---------------------------------------------------------------------------
// Usage API
// ---------------------------------------------------------------------------

interface UsageData {
  plan: string;
  requestsUsed: number;
  requestsLimit: number | null;
  remainingRequests: number | null;
  resetsInMinutes: number | null;
  concurrent: number;
  concurrentLimit: number | null;
  tokensIn: number;
  tokensOut: number;
}

async function fetchUsage(apiKey: string): Promise<UsageData | null> {
  try {
    const res = await fetch(USAGE_URL, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    const plan = data.plan?.display_name || data.plan?.slug || "Unknown";
    const limits = data.limits ?? {};
    const usage = data.usage ?? {};
    const window = data.window ?? {};

    return {
      plan,
      requestsUsed: usage.requests_in_window ?? 0,
      requestsLimit: limits.requests?.limit ?? null,
      remainingRequests: usage.remaining_requests ?? null,
      resetsInMinutes: window.remaining_minutes ?? null,
      concurrent: usage.concurrent_sessions ?? 0,
      concurrentLimit: limits.concurrency?.limit ?? null,
      tokensIn: usage.tokens_in ?? 0,
      tokensOut: usage.tokens_out ?? 0,
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function fmtDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// Shared state between status bar helpers
let lastApiKey: string | null = null;

async function updateUsageStatus(ctx: any, tps?: string, ttft?: string): Promise<void> {
  const theme = ctx.ui.theme;
  const apiKey = lastApiKey;
  if (!apiKey) return;

  const usage = await fetchUsage(apiKey);
  if (!usage) return;

  const perfParts: string[] = [];
  if (tps) perfParts.push(`T/S:${tps}`);
  if (ttft) perfParts.push(`TTFT:${ttft}`);
  const perfStr = perfParts.length > 0 ? perfParts.join(" │ ") + " │ " : "";

  const reqPart = usage.requestsLimit !== null
    ? `${usage.requestsUsed}/${usage.requestsLimit}`
    : `${usage.requestsUsed}`;
  const resetPart = usage.resetsInMinutes !== null ? ` ⟳${usage.resetsInMinutes}m` : "";

  ctx.ui.setWidget(
    "umans",
    [theme.fg("dim", `Umans ${perfStr}${reqPart}${resetPart}`)],
    { placement: "belowEditor" },
  );
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  // --- Provider registration ---
  pi.registerProvider("umans", {
    baseUrl: "https://api.code.umans.ai/v1",
    api: "openai-completions",
    apiKey: "UMANS_API_KEY",
    authHeader: true,
    models,
    oauth: {
      name: "Umans AI (API Key)",
      login: loginUmans,
      refreshToken: refreshUmansToken,
      getApiKey,
    },
  });

  // Thinking is driven by `thinkingFormat: "deepseek"` + a per-model `thinkingLevelMap`
  // (built in buildReasoningConfig from each model's advertised `reasoning.levels`).
  // For level-capable models (e.g. GLM 5.2: none/high/max) pi sends
  //   thinking: { type: "enabled" } + reasoning_effort: "<mapped level>"
  // so pi's "xhigh" reaches Umans as `reasoning_effort: "max"`. For on/off-only models
  // (e.g. Kimi) supportsReasoningEffort is false, so pi only sends the thinking field.
  // No stripping: the Umans endpoint accepts reasoning_effort for GLM (none/high/max)
  // and ignores it for Kimi without error.
  //
  // Also sanitize conversation history: ensure every tool_calls entry has a matching
  // tool result message. Context compaction can drop tool result messages while keeping
  // the assistant message that made the tool call, causing a 400 error from the API:
  //   "an assistant message with 'tool_calls' must be followed by tool messages
  //    responding to each 'tool_call_id'"
  pi.on("before_provider_request", async (event) => {
    const p = event.payload as Record<string, any>;
    const model: string = p.model ?? "";
    if (!model.startsWith("umans-")) return;

    // --- Vision Bridge: describe images in PARALLEL for text-only models ---
    if (isTextOnlyUmansModel(model) && Array.isArray(p.messages)) {
      const imageRefs: Array<{ msg: any; index: number; imageUrl: string }> = [];
      for (const msg of p.messages) {
        if (!Array.isArray(msg.content)) continue;
        for (let i = 0; i < msg.content.length; i++) {
          const part = msg.content[i];
          if (part?.type !== "image_url") continue;
          const imageUrl = part.image_url?.url ?? part.image_url;
          if (typeof imageUrl !== "string") continue;
          imageRefs.push({ msg, index: i, imageUrl });
        }
      }
      if (imageRefs.length > 0) {
        const results = await mapWithConcurrency(
          imageRefs,
          async ({ imageUrl }) => {
            try {
              return await describeImageViaGateway(imageUrl);
            } catch (err) {
              return { description: `Image was present but vision bridge failed: ${String(err).slice(0, 80)}`, cached: false };
            }
          },
          3,
        );
        for (let j = 0; j < imageRefs.length; j++) {
          const { msg, index } = imageRefs[j];
          const { description } = results[j];
          msg.content[index] = {
            type: "text",
            text: `[Image described by ${VISION_BRIDGE_MODEL}: ${description}]`,
          };
        }
      }
      // Silent: no log on success
    }

    // --- Sanitize orphaned tool_calls in OpenAI-format messages ---
    const messages = p.messages;
    if (!Array.isArray(messages) || messages.length === 0) return;

    // Collect all tool_call IDs from assistant messages
    const toolCallIds = new Set<string>();
    for (const msg of messages) {
      if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          if (tc.id) toolCallIds.add(tc.id);
        }
      }
    }

    // Collect all tool_call_ids from tool result messages
    const toolResultIds = new Set<string>();
    for (const msg of messages) {
      if (msg.role === "tool" && msg.tool_call_id) {
        toolResultIds.add(msg.tool_call_id);
      }
    }

    // Find orphaned IDs (assistant made a tool_call but no tool result exists)
    const orphanedIds = [...toolCallIds].filter((id) => !toolResultIds.has(id));
    if (orphanedIds.length === 0) return;

    console.warn(
      `[pi-provider-umans] Found ${orphanedIds.length} orphaned tool_call(s) without tool result: ${orphanedIds.join(", ")}`,
    );

    // Insert synthetic tool result messages after the assistant message that made each call
    const newMessages = [...messages];
    let insertOffset = 0;

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role !== "assistant" || !Array.isArray(msg.tool_calls)) continue;

      const orphanedCalls = msg.tool_calls.filter((tc: any) =>
        orphanedIds.includes(tc.id),
      );
      if (orphanedCalls.length === 0) continue;

      // Insert synthetic tool results right after this assistant message
      const insertIdx = i + insertOffset + 1;
      const syntheticResults = orphanedCalls.map((tc: any) => ({
        role: "tool",
        tool_call_id: tc.id,
        content: "[tool result was lost during context compaction]",
      }));

      newMessages.splice(insertIdx, 0, ...syntheticResults);
      insertOffset += orphanedCalls.length;
    }

    p.messages = newMessages;
    return p;
  });

  // --- Status bar: usage + performance ---
  let turnStartTime = 0;
  let firstTokenTime = 0;

  // Show usage on session start
  pi.on("session_start", async (_event, ctx) => {
    const apiKey = await ctx.modelRegistry.getApiKeyForProvider("umans").catch(() => undefined);
    if (apiKey) {
      lastApiKey = apiKey;
      await updateUsageStatus(ctx, undefined, undefined);
    } else {
      const theme = ctx.ui.theme;
      ctx.ui.setWidget("umans", [theme.fg("dim", "Umans: /login umans")], { placement: "belowEditor" });
    }
  });

  // Track turn timing
  pi.on("turn_start", async (_event, _ctx) => {
    turnStartTime = Date.now();
    firstTokenTime = 0;
  });

  // Track first token (TTFT) — just record the time, no separate status
  pi.on("message_update", async (event, _ctx) => {
    if (firstTokenTime === 0 && turnStartTime > 0) {
      const msg = event.message as AssistantMessage;
      if (msg.role === "assistant" && msg.content?.length > 0) {
        firstTokenTime = Date.now();
      }
    }
  });

  // On turn end, compute TPS and refresh usage
  pi.on("turn_end", async (event, ctx) => {
    const theme = ctx.ui.theme;
    const msg = event.message as AssistantMessage;

    if (msg.role !== "assistant" || turnStartTime === 0) return;

    const elapsed = Date.now() - turnStartTime;
    const outputTokens = msg.usage?.output ?? 0;
    const tps =
      elapsed > 0 && outputTokens > 0 ? (outputTokens / (elapsed / 1000)).toFixed(0) : "—";
    const ttft = firstTokenTime > 0 ? fmtDuration(firstTokenTime - turnStartTime) : "—";

    // Resolve API key from model registry (covers env var and OAuth)
    let apiKey = await ctx.modelRegistry.getApiKeyForProvider("umans").catch(() => undefined) || lastApiKey;
    let usageStr = "";

    if (apiKey) {
      lastApiKey = apiKey;
      const usage = await fetchUsage(apiKey);
      if (usage) {
        const reqPart =
          usage.requestsLimit !== null
            ? `${usage.requestsUsed}/${usage.requestsLimit}`
            : `${usage.requestsUsed}`;
        const resetPart =
          usage.resetsInMinutes !== null
            ? ` ⟳${usage.resetsInMinutes}m`
            : "";
        usageStr = ` │ ${reqPart}${resetPart}`;
      }
    }

    const perfStr = usageStr ? `T/S:${tps} │ TTFT:${ttft}${usageStr}` : `T/S:${tps} │ TTFT:${ttft}`;
    ctx.ui.setWidget(
      "umans",
      [theme.fg("dim", `Umans ${perfStr}`)],
      { placement: "belowEditor" },
    );
  });
}
