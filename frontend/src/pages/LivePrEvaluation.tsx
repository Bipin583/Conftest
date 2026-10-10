import { useState } from "react";
import { useSelectMutation } from "../api/client";
import { PageHeader, Card } from "../components/ui";
import { ArtifactBlocker, Loading } from "../components/states";
import Chart from "../components/Chart";
import { SERIES, STATUS } from "../lib/plot";
import type { SelectResponse } from "../api/types";

function DecisionBadge({ result }: { result: SelectResponse }) {
  const abstain = result.abstained;
  const color = abstain ? STATUS.warning : STATUS.good;
  return (
    <div className="flex flex-wrap items-center gap-4">
      <span className="rounded-lg px-3 py-1 text-sm font-semibold" style={{ background: `${color}22`, color }}>
        {result.decision_mode}
      </span>
      <span className="text-sm text-ink-secondary">
        {result.selected_count}/{result.total_count} tests · {result.test_reduction_pct.toFixed(1)}% reduction · top
        confidence {(result.top_confidence * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function RankedChart({ result }: { result: SelectResponse }) {
  const tests = [...result.ranked_tests].slice(0, 15).reverse();
  return (
    <Chart
      height={Math.max(240, tests.length * 26)}
      data={[
        {
          type: "bar",
          orientation: "h",
          y: tests.map((t) => t.test_id.split("::").pop() ?? t.test_id),
          x: tests.map((t) => t.calibrated_confidence),
          marker: {
            color: tests.map((t) => (t.is_selected ? SERIES[0] : "#898781")),
            line: { color: "#1a1a19", width: 1 },
          },
          hovertemplate: "%{y}<br>confidence %{x:.4f}<extra></extra>",
        },
      ]}
      layout={{
        title: { text: "Per-test calibrated failure confidence", font: { size: 14 } },
        xaxis: { title: { text: "calibrated confidence" } },
        yaxis: { automargin: true, tickfont: { color: "#898781" } },
        margin: { l: 20, r: 20, t: 36, b: 40 },
      }}
    />
  );
}

export default function LivePrEvaluation() {
  const [repo, setRepo] = useState("local/sample-app");
  const [message, setMessage] = useState("refactor auth token validation");
  const [files, setFiles] = useState("src/app/auth.py");
  const [budget, setBudget] = useState(0.25);
  const mutation = useSelectMutation();

  const run = () => {
    mutation.mutate({
      repository_name: repo,
      commit_sha: "HEAD",
      commit_message: message,
      changed_files: files
        .split("\n")
        .map((f) => f.trim())
        .filter(Boolean)
        .map((file_path) => ({ file_path, change_type: "modified" })),
      budget_ratio: budget,
    });
  };

  return (
    <div>
      <PageHeader title="🚀 Live PR Evaluation" subtitle="Score a changed-file set and see the SELECT / ABSTAIN decision." />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <Card title="Pull request">
            <label className="mb-3 block text-sm">
              <span className="text-muted">Repository</span>
              <input className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 p-2" value={repo} onChange={(e) => setRepo(e.target.value)} />
            </label>
            <label className="mb-3 block text-sm">
              <span className="text-muted">Commit message</span>
              <input className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 p-2" value={message} onChange={(e) => setMessage(e.target.value)} />
            </label>
            <label className="mb-3 block text-sm">
              <span className="text-muted">Changed files (one per line)</span>
              <textarea className="mt-1 h-28 w-full rounded-lg border border-white/10 bg-white/5 p-2 font-mono text-xs" value={files} onChange={(e) => setFiles(e.target.value)} />
            </label>
            <label className="mb-4 block text-sm">
              <span className="text-muted">Budget ratio: {budget.toFixed(2)}</span>
              <input type="range" min={0.01} max={1} step={0.01} value={budget} onChange={(e) => setBudget(Number(e.target.value))} className="mt-1 w-full" />
            </label>
            <button onClick={run} disabled={mutation.isPending} className="w-full rounded-lg bg-series-1 px-4 py-2 font-semibold text-white disabled:opacity-50">
              {mutation.isPending ? "Scoring…" : "Run selection"}
            </button>
          </Card>
        </div>
        <div className="lg:col-span-3">
          <Card title="Decision">
            {mutation.isIdle && <p className="text-ink-secondary">Submit a PR to see the selection decision.</p>}
            {mutation.isPending && <Loading label="Scoring tests…" />}
            {mutation.isError && <ArtifactBlocker error={mutation.error} />}
            {mutation.isSuccess && (
              <div className="space-y-4">
                <DecisionBadge result={mutation.data} />
                <ul className="space-y-1 text-sm text-ink-secondary">
                  {mutation.data.reasons.map((r, i) => (
                    <li key={i}>• {r}</li>
                  ))}
                </ul>
                <RankedChart result={mutation.data} />
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
