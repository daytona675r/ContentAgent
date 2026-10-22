import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, DemoRun, Scenario } from "./api";
import "./style.css";

function App() {
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

function Workflow({ result }: { result: DemoRun }) {
  return <>
    <section className="context panel"><h2>Business context</h2><p>{result.scenario.context}</p></section>
    <section className="panel"><h2>Initial platform drafts</h2><div className="grid">{Object.entries(result.initial_drafts).map(([platform, text]) => <article key={platform}><h3>{platform}</h3><p>{text}</p></article>)}</div></section>
    <section className="panel"><h2>Persona evaluation</h2><div className="grid">{Object.entries(result.persona_scores).map(([persona, score]) => <article key={persona}><h3>{persona}</h3><strong>{score.toFixed(1)} / 10</strong></article>)}</div><p>{result.persona_feedback}</p></section>
    <section className="panel"><h2>Refinement history</h2>{result.refinement_history.map((step) => <div className="step" key={step.stage}><span>{step.stage}</span><p>{step.content}</p></div>)}</section>
    <section className="panel accepted"><h2>Final accepted content</h2>{Object.entries(result.final_accepted).map(([platform, text]) => <article key={platform}><h3>{platform}</h3><p>{text}</p></article>)}<small>{result.token_cost.tokens} simulated tokens · ${result.token_cost.estimated_usd.toFixed(2)} example cost</small></section>
  </>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
