import { useAnalytics } from "../api/client";
import { PageHeader, Card, MetricCard } from "../components/ui";
import { QueryGate } from "../components/states";
import type { AnalyticsResponse } from "../api/types";

function DecisionsTable({ data }: { data: AnalyticsResponse }) {
  if (data.recent_decisions.length === 0) {
    return (
      <p className="text-ink-secondary">
        No decisions recorded yet. Run a selection on the Live PR Evaluation page, or point a GitHub
        webhook at <code>/api/v1/github/webhook</code> — every decision is persisted and shown here.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead className="text-left text-muted">
          <tr>
            <th className="py-1 pr-4">Commit</th>
            <th className="pr-4">Mode</th>
            <th>Selected</th>
            <th>Reduction %</th>
            <th>Uncertainty</th>
            <th>When</th>
          </tr>
        </thead>
        <tbody>
          {data.recent_decisions.map((d, i) => (
            <tr key={i} className="border-t border-white/10">
              <td className="py-1.5 pr-4 font-mono text-xs">{d.commit_sha.slice(0, 12)}</td>
              <td className="pr-4">
                <span className={d.abstained ? "text-warning" : "text-good"}>{d.mode}</span>
              </td>
              <td>{d.selected_count}/{d.total_count}</td>
              <td>{d.test_reduction_pct.toFixed(1)}</td>
              <td>{d.uncertainty.toFixed(4)}</td>
              <td className="text-muted">{new Date(d.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Analytics() {
  const query = useAnalytics();
  return (
    <div>
      <PageHeader
        title="📈 Analytics & Telemetry"
        subtitle="Persisted decisions from live evaluations and signed GitHub webhook events."
      />
      <QueryGate query={query}>
        {(data) => (
          <>
            <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Total decisions" value={String(data.total_decisions)} note={`${data.total_repositories} repo(s), ${data.total_commits_evaluated} commit(s)`} />
              <MetricCard label="Safe full-suite abstentions" value={String(data.total_safe_abstentions)} note={`${data.total_selective_fast_mode} fast-mode selections`} />
              <MetricCard label="Avg test reduction" value={`${data.average_test_reduction_pct.toFixed(1)}%`} />
              <MetricCard label="Avg epistemic uncertainty" value={data.average_uncertainty.toFixed(4)} note={`${data.verified_outcomes} verified / ${data.unverified_outcomes} unverified outcomes`} />
            </div>
            <Card title="Recent decisions">
              <DecisionsTable data={data} />
            </Card>
          </>
        )}
      </QueryGate>
    </div>
  );
}
