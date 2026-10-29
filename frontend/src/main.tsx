import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, DemoRun, Scenario } from "./api";
import "./style.css";

export function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selected, setSelected] = useState("customer-context");
  const [result, setResult] = useState<DemoRun>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    api.scenarios().then(setScenarios).catch((reason: Error) => setError(reason.message));
  }, []);

  async function runDemo() {
    setLoading(true);
    setError(undefined);
    try {
      setResult(await api.runDemo(selected));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to run demo");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">PULSE · applied AI workflow engineering</p>
        <h1>From context to accepted content.</h1>
        <p className="lede">Inspect the deterministic workflow behind generation, evaluation and refinement.</p>
        <span className="badge">Demo / replayed data · no provider calls</span>
      </header>
      <section className="controls">
        <label htmlFor="scenario">Scenario</label>
        <select id="scenario" value={selected} onChange={(event) => setSelected(event.target.value)}>
          {scenarios.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.label}</option>)}
        </select>
        <button onClick={runDemo} disabled={loading}>{loading ? "Running…" : "Run workflow"}</button>
      </section>
      {error && <p className="error">{error}</p>}
      {result && <Workflow result={result} />}
    </main>
  );
}

export function Workflow({ result }: { result: DemoRun }) {
  return <>
    <section className="context panel"><h2>Business context</h2><p>{result.scenario.context}</p><small>Fictional brief · authored replay data</small></section>
    <section className="panel"><h2>Platform progression</h2><p className="muted">Each example shows the initial draft, the strategist&apos;s actionable critique, and the accepted refinement from the real graph trace.</p><div className="platforms">{Object.keys(result.initial_drafts).map((platform) => {
      const history = result.refinement_history[platform] ?? [];
      const evaluations = result.evaluation_history[platform] ?? [];
      const initialEvaluation = evaluations[0];
      const acceptedEvaluation = evaluations[evaluations.length - 1];
      return <article className="platform" key={platform}><h3>{platform}</h3><div className="progression"><div><span>Initial draft · sample</span><p>{history[0]}</p></div><div><span>Evaluation · strategist</span><p>{initialEvaluation?.feedback}</p><strong>{initialEvaluation?.score.toFixed(1)} / 10</strong></div><div><span>Refined post · sample</span><p>{result.final_accepted[platform]}</p></div><div><span>Accepted score · replay</span><strong>{acceptedEvaluation?.score.toFixed(1)} / 10</strong><p>{acceptedEvaluation?.feedback}</p></div></div></article>;
    })}</div></section>
    <section className="panel accepted"><h2>Workflow trace</h2><p>{result.persona_feedback}</p><small>{result.token_cost.tokens} simulated tokens · ${result.token_cost.estimated_usd.toFixed(2)} example cost · no live AI calls</small></section>
  </>;
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<StrictMode><App /></StrictMode>);
