import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Workflow, defaultScenarioId } from "./main";

const result = {
  demo: true,
  notice: "Replayed sample data",
  scenario: { id: "x", label: "Fictional sample", context: "Fictional brief" },
  initial_drafts: { twitter: "Initial draft" },
  persona_scores: { twitter: 9.2 },
  persona_feedback: "Accepted feedback",
  refinement_history: { twitter: ["Initial draft", "Refined draft"] },
  evaluation_history: { twitter: [{ stage: "initial", score: 6.8, feedback: "Name the audience" }, { stage: "accepted", score: 9.2, feedback: "Accepted" }] },
  final_accepted: { twitter: "Refined draft" },
  token_cost: { tokens: 0, estimated_usd: 0 },
};

describe("scenario initialization", () => {
  it("selects the first actual API scenario instead of a removed placeholder ID", () => {
    expect(defaultScenarioId([
      { id: "onboarding-saas", label: "Onboarding", context: "Fictional SaaS" },
      { id: "repairable-commuter", label: "Commuter", context: "Fictional light" },
    ])).toBe("onboarding-saas");
    expect(defaultScenarioId([])).toBe("");
  });
});

describe("workflow presentation", () => {
  it("keeps platform progression and replay labels visible", () => {
    const html = renderToStaticMarkup(<Workflow result={result} />);
    expect(html).toContain("Initial draft · sample");
    expect(html).toContain("Name the audience");
    expect(html).toContain("Refined post · sample");
    expect(html).toContain("no live AI calls");
  });
});
