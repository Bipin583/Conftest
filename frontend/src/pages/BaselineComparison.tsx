import { useBaseline } from "../api/client";
import { PageHeader, Card, ProvenanceBanner } from "../components/ui";
import { QueryGate } from "../components/states";
import Chart from "../components/Chart";
import { SERIES } from "../lib/plot";
import type { BaselineRow } from "../api/types";

const isConf = (s: string | null) => (s ?? "").includes("ConfTest");

function BarByMetric({ rows, metric, title, axis }: { rows: BaselineRow[]; metric: keyof BaselineRow; title: string; axis: string }) {
  const valid = rows.filter((r) => r[metric] !== null);
  return (
    <Chart
      height={Math.max(240, valid.length * 34)}
      data={[
        {
          type: "bar",
          orientation: "h",
          y: valid.map((r) => r.strategy ?? ""),
          x: valid.map((r) => r[metric] as number),
          marker: {
            color: valid.map((r) => (isConf(r.strategy) ? SERIES[0] : "#898781")),
            line: { color: "#1a1a19", width: 1 },
          },
          hovertemplate: `%{y}<br>${axis} %{x:.1f}%<extra></extra>`,
        },
      ]}
      layout={{ title: { text: title, font: { size: 14 } }, xaxis: { title: { text: axis } }, yaxis: { automargin: true, tickfont: { color: "#898781" } }, margin: { l: 20, r: 20, t: 36, b: 40 } }}
    />
  );
}

function Table({ rows }: { rows: BaselineRow[] }) {
  const fmt = (v: number | null) => (v === null ? "—" : v.toFixed(1));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead className="text-left text-muted">
          <tr>
            <th className="py-1 pr-4">Strategy</th>
            <th>Recall %</th>
            <th>Time saved %</th>
            <th>Test cut %</th>
            <th>Missed %</th>
            <th>Abstain %</th>
            <th>Escaped</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.strategy} className={`border-t border-white/10 ${isConf(r.strategy) ? "text-series-1" : ""}`}>
              <td className="py-1.5 pr-4">{r.strategy}</td>
              <td>{fmt(r.failure_recall_pct)}</td>
              <td>{fmt(r.time_reduction_pct)}</td>
              <td>{fmt(r.test_reduction_pct)}</td>
              <td>{fmt(r.missed_failure_pct)}</td>
              <td>{fmt(r.abstention_rate_pct)}</td>
              <td>{r.escaped_commits === null ? "—" : r.escaped_commits}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function BaselineComparison() {
  const query = useBaseline();
  return (
    <div>
      <PageHeader title="📊 Baseline Comparison" subtitle="ConfTest against classical RTS strategies on the measured benchmark." />
      <QueryGate query={query}>
        {(data) => (
          <>
            <ProvenanceBanner provenance={data.provenance} />
            <Card title="Measured benchmark table">
              <Table rows={data.rows} />
            </Card>
            <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
              <Card>
                <BarByMetric rows={data.rows} metric="failure_recall_pct" title="Failure recall by strategy" axis="recall %" />
              </Card>
              <Card>
                <BarByMetric rows={data.rows} metric="time_reduction_pct" title="Time saved by strategy" axis="time reduction %" />
              </Card>
            </div>
          </>
        )}
      </QueryGate>
    </div>
  );
}
