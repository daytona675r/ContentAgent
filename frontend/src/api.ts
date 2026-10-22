export type Scenario = { id: string; label: string; context: string };
export type DemoRun = {
  demo: boolean;
  notice: string;
  scenario: Scenario;
  initial_drafts: Record<string, string>;
  persona_scores: Record<string, number>;
  persona_feedback: string;
  refinement_history: Array<{ stage: string; content: string }>;
  final_accepted: Record<string, string>;
  token_cost: { tokens: number; estimated_usd: number };
};

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const baseUrl = configuredBaseUrl.startsWith("http")
  ? configuredBaseUrl
  : `https://${configuredBaseUrl}`;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export const api = {
  scenarios: () => request<Scenario[]>("/api/demo/scenarios"),
  runDemo: (scenarioId: string) =>
    request<DemoRun>("/api/demo/run", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    }),
};
