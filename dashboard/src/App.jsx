import React, { useEffect, useMemo, useState } from "react";
import RidgePcaView from "./RidgePcaView.jsx";

const fmt = (value, digits = 5) =>
  Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "n/a";

const short = (value, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "n/a";

const plain = (value, digits = 4) =>
  Number.isFinite(value) ? value.toFixed(digits) : value === null || value === undefined ? "n/a" : String(value);

function contributionForCandidate(run, snapshot, candidate) {
  const rows = run.features.map((feature, index) => {
    const raw = candidate.xMean[index];
    const value = Number.isFinite(raw) ? raw : snapshot.imputeMeans[index];
    const z = (value - snapshot.mu[index]) / snapshot.sd[index];
    const contribution = z * snapshot.weights[index];
    return { feature, value, z, weight: snapshot.weights[index], contribution };
  });
  const total = rows.reduce((sum, row) => sum + row.contribution, snapshot.base);
  return { rows, total };
}

function groupContributions(rows, keep = 8) {
  const sorted = [...rows].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const head = sorted.slice(0, keep);
  const rest = sorted.slice(keep);
  const restValue = rest.reduce((sum, row) => sum + row.contribution, 0);
  return rest.length ? [...head, { feature: "other features", contribution: restValue }] : head;
}

function Waterfall({ run, snapshot, candidate }) {
  const data = useMemo(() => {
    if (!run || !snapshot || !candidate) return null;
    const { rows, total } = contributionForCandidate(run, snapshot, candidate);
    const bars = [
      { feature: "base", contribution: snapshot.base, kind: "base" },
      ...groupContributions(rows),
      { feature: "prediction", contribution: total, kind: "final" },
    ];
    let cumulative = 0;
    const laidOut = bars.map((bar) => {
      if (bar.kind === "final") {
        return { ...bar, start: 0, end: bar.contribution };
      }
      const start = cumulative;
      const end = cumulative + bar.contribution;
      cumulative = end;
      return { ...bar, start, end };
    });
    return { bars: laidOut, total, rows };
  }, [run, snapshot, candidate]);

  if (!data) return <main className="stage empty">No run selected</main>;

  const width = 980;
  const left = 210;
  const right = 36;
  const top = 40;
  const rowH = 32;
  const chartW = width - left - right;
  const height = top + data.bars.length * rowH + 32;
  const extents = data.bars.flatMap((bar) => [bar.start, bar.end, 0]);
  let min = Math.min(...extents);
  let max = Math.max(...extents);
  const pad = Math.max((max - min) * 0.12, 0.001);
  min -= pad;
  max += pad;
  const x = (value) => left + ((value - min) / (max - min || 1)) * chartW;
  const zeroX = x(0);

  return (
    <main className="stage">
      <div className="vizHeader">
        <div>
          <h1>{candidate.id}</h1>
        </div>
        <div className="metricStrip">
          <Metric label="Predicted lift" value={fmt(data.total)} tone={data.total >= 0 ? "good" : "bad"} />
          <Metric label="Measured lift" value={fmt(candidate.trueMean)} tone={candidate.trueMean >= 0 ? "good" : "bad"} />
          <Metric label="Step" value={`${snapshot.t}/${run.rowCount}`} />
          <Metric label="Train RMSE" value={short(snapshot.trainRmse, 5)} />
        </div>
      </div>

      <div className="chartPanel">
        <svg className="waterfall" viewBox={`0 0 ${width} ${height}`} role="img">
          <line x1={zeroX} x2={zeroX} y1={18} y2={height - 18} className="zeroLine" />
          {data.bars.map((bar, index) => {
            const y = top + index * rowH;
            const x1 = x(Math.min(bar.start, bar.end));
            const x2 = x(Math.max(bar.start, bar.end));
            const w = Math.max(x2 - x1, 2);
            const positive = bar.end >= bar.start;
            const cls = bar.kind === "final" ? "bar final" : bar.kind === "base" ? "bar base" : positive ? "bar pos" : "bar neg";
            return (
              <g key={`${bar.feature}-${index}`}>
                <text x={24} y={y + 22} className="featureLabel">{bar.feature}</text>
                <rect x={x1} y={y + 6} width={w} height={18} rx={3} className={cls} />
                <line x1={x(bar.end)} x2={x(bar.end)} y1={y + 24} y2={y + rowH + 4} className="connector" />
                <text x={bar.end >= 0 ? x2 + 8 : x1 - 8} y={y + 20} textAnchor={bar.end >= 0 ? "start" : "end"} className="valueLabel">
                  {fmt(bar.contribution)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </main>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className={`metric ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatTable({ rows }) {
  return (
    <div className="statGrid">
      {rows.map(([label, value, tone]) => (
        <div className={`statCell ${tone || ""}`} key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function DataExplorer({ run, candidate }) {
  const [rowIndex, setRowIndex] = useState(0);
  const rows = candidate?.rows || [];
  const selected = rows[Math.min(rowIndex, Math.max(rows.length - 1, 0))];

  useEffect(() => {
    setRowIndex(0);
  }, [candidate?.id]);

  if (!candidate) return <main className="stage empty">No candidate selected</main>;

  const summary = selected?.rewardSummary || {};
  const rawText = selected ? JSON.stringify(selected.raw, null, 2) : "{}";
  const trajectoryLabel = candidate.hasRawTrajectories ? "Raw rollout trajectories" : "Branch-label rows";

  return (
    <main className="stage explorerStage">
      <div className="vizHeader">
        <div>
          <h1>{candidate.id}</h1>
        </div>
        <div className="metricStrip">
          <Metric label="Rows" value={String(rows.length)} />
          <Metric label="Prompts" value={String(candidate.promptIds?.length || 0)} />
          <Metric label="Mean lift" value={fmt(candidate.trueMean)} tone={candidate.trueMean >= 0 ? "good" : "bad"} />
          <Metric label="Mean n_contrib" value={short(candidate.nContribMean, 2)} />
        </div>
      </div>

      <div className="explorerGrid">
        <section className="dataPanel">
          <div className="panelHead">
            <div>
              <h3>Candidate Cohort</h3>
            </div>
            <span className="pill">{candidate.noise === null || candidate.noise === undefined ? candidate.kind : `noise ${candidate.noise}`}</span>
          </div>
          <StatTable
            rows={[
              ["Measured lift", fmt(candidate.trueMean), candidate.trueMean >= 0 ? "good" : "bad"],
              ["Lift range", `${fmt(candidate.trueMin)} to ${fmt(candidate.trueMax)}`],
              ["KL drift mean", plain(candidate.klMean, 5)],
              ["Target similarity", plain(candidate.targetSimilarity, 4)],
            ]}
          />
          <h4>Prompts</h4>
          <div className="promptList">
            {(candidate.prompts || candidate.promptIds || []).map((item) => {
              const prompt = typeof item === "string" ? null : item.prompt;
              const id = typeof item === "string" ? item : item.id;
              return (
                <article className={`promptCard ${prompt ? "" : "missing"}`} key={id}>
                  <div className="promptCardHead">
                    <code>{id}</code>
                    {item.answer !== undefined && item.answer !== null && <span>answer {String(item.answer)}</span>}
                  </div>
                  {prompt ? (
                    <>
                      <p>{prompt}</p>
                    </>
                  ) : (
                    <p>Prompt text is not present in the local data cache for this ID.</p>
                  )}
                </article>
              );
            })}
          </div>
        </section>

        <section className="dataPanel">
          <div className="panelHead">
            <div>
              <h3>Per-Seed Branches</h3>
              <p className="sectionMeta">{trajectoryLabel}</p>
            </div>
            {!candidate.hasRawTrajectories && <span className="pill muted">aggregate only</span>}
          </div>
          {!candidate.hasRawTrajectories && (
            <p className="note">
              Raw completion trajectories were not found for this run. These cards show the stored branch-label rows:
              one candidate update repeat per seed, with aggregate rollout statistics.
            </p>
          )}
          <div className="trajectoryList">
            {rows.map((row, index) => (
              <button
                key={`${candidate.id}-${row.index}`}
                className={`trajectoryCard ${index === rowIndex ? "selected" : ""}`}
                onClick={() => setRowIndex(index)}
              >
                <div>
                  <strong>seed {row.seed ?? "n/a"}</strong>
                  <span>chain {row.chainId ?? 0} · anchor {row.anchorIndex ?? 0}</span>
                </div>
                <div>
                  <b className={Number(row.liftNll) >= 0 ? "green" : "red"}>{fmt(row.liftNll)}</b>
                  <span>{plain(row.nContrib, 0)} contrib</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="dataPanel wide">
          <div className="panelHead">
            <div>
              <h3>Labels And Rollout Summary</h3>
            </div>
            <span className="pill">seed {selected?.seed ?? "n/a"}</span>
          </div>
          <StatTable
            rows={[
              ["lift_nll", fmt(selected?.liftNll), Number(selected?.liftNll) >= 0 ? "good" : "bad"],
              ["lift_acc", fmt(selected?.liftAcc)],
              ["utility", fmt(selected?.utility), Number(selected?.utility) >= 0 ? "good" : "bad"],
              ["kl_drift", plain(selected?.klDrift, 5)],
              ["mean_reward", plain(selected?.meanReward, 4)],
              ["n_contrib", plain(selected?.nContrib, 0)],
              ["rollout_count", plain(selected?.rolloutCount, 0)],
              ["wall_clock_s", plain(selected?.wallClockS, 1)],
            ]}
          />
          <h4>Reward Summary</h4>
          <div className="featureGrid">
            {Object.entries(summary).map(([key, value]) => (
              <div className="featureChip" key={key}>
                <span>{key}</span>
                <strong>{plain(value, 4)}</strong>
              </div>
            ))}
          </div>
        </section>

        <details className="dataPanel wide jsonPanel">
          <summary>Raw JSON</summary>
          <pre>{rawText}</pre>
        </details>
      </div>
    </main>
  );
}

function Sidebar({ run, candidateId, setCandidateId, visualization, setVisualization }) {
  return (
    <aside className="sidebar">
      <div className="sideBlock">
        <label>Visualization</label>
        <button
          className={`vizButton ${visualization === "waterfall" ? "active" : ""}`}
          onClick={() => setVisualization("waterfall")}
        >
          Waterfall
        </button>
        <button
          className={`vizButton ${visualization === "explorer" ? "active" : ""}`}
          onClick={() => setVisualization("explorer")}
        >
          Data Explorer
        </button>
        <button
          className={`vizButton ${visualization === "ridgePca" ? "active" : ""}`}
          onClick={() => setVisualization("ridgePca")}
        >
          Ridge Gradient 3D
        </button>
      </div>
      <div className="sideBlock grow">
        <label>Candidate</label>
        <div className="candidateList">
          {run?.candidates.map((candidate) => (
            <button
              key={candidate.id}
              className={`candidateButton ${candidate.id === candidateId ? "selected" : ""}`}
              onClick={() => setCandidateId(candidate.id)}
            >
              <span>{candidate.id}</span>
              <small>
                {candidate.noise === null || candidate.noise === undefined ? candidate.kind : `noise ${candidate.noise}`}
              </small>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [runId, setRunId] = useState("");
  const [step, setStep] = useState(1);
  const [candidateId, setCandidateId] = useState("");
  const [visualization, setVisualization] = useState("waterfall");

  useEffect(() => {
    fetch("/tap_dashboard_data.json")
      .then((response) => response.json())
      .then((payload) => {
        setData(payload);
        const first = payload.runs[0];
        setRunId(first.id);
        setStep(first.rowCount);
        setCandidateId(first.candidates[0]?.id || "");
      });
  }, []);

  const run = useMemo(() => data?.runs.find((item) => item.id === runId), [data, runId]);
  const snapshot = run?.snapshots[Math.max(0, Math.min(step, run.rowCount) - 1)];
  const candidate = run?.candidates.find((item) => item.id === candidateId) || run?.candidates[0];
  const handleStepInput = (event) => setStep(Number(event.target.value));

  useEffect(() => {
    if (!run) return;
    setStep(run.rowCount);
    setCandidateId(run.candidates[0]?.id || "");
  }, [runId]);

  if (!data || !run) {
    return <div className="loading">Loading TAP dashboard</div>;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="appLabel">TAP ridge dashboard</p>
          <h2>{run.name}</h2>
        </div>
        <div className="runPicker">
          <label htmlFor="run-select">Training run</label>
          <select id="run-select" value={runId} onChange={(event) => setRunId(event.target.value)}>
            {data.runs.map((item) => (
              <option key={item.id} value={item.id}>{item.name}</option>
            ))}
          </select>
        </div>
      </header>

      <div className="workspace">
        {visualization === "ridgePca" ? (
          <RidgePcaView run={run} step={step} />
        ) : visualization === "explorer" ? (
          <DataExplorer run={run} candidate={candidate} />
        ) : (
          <Waterfall run={run} snapshot={snapshot} candidate={candidate} />
        )}
        <Sidebar
          run={run}
          candidateId={candidate?.id || ""}
          setCandidateId={setCandidateId}
          visualization={visualization}
          setVisualization={setVisualization}
        />
      </div>

      <footer className="stepbar">
        <div className="stepMeta">
          <span>t</span>
          <strong>{step}</strong>
          <span>{run.rowCount} labels</span>
        </div>
        <input
          type="range"
          min="1"
          max={run.rowCount}
          value={step}
          onChange={handleStepInput}
          onInput={handleStepInput}
        />
      </footer>
    </div>
  );
}
