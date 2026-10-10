import { useCalibration } from "../api/client";
import { PageHeader, Card } from "../components/ui";
import { QueryGate } from "../components/states";
import Chart from "../components/Chart";
import { SERIES } from "../lib/plot";
import type { CalibrationResponse } from "../api/types";

function MetricsTable({ data }: { data: CalibrationResponse }) {
  const rows = [{ ...data.uncalibrated }];
  if (data.calibrated) rows.push({ ...data.calibrated });
  return (
    <table className="w-full text-sm tabular-nums">
      <thead className="text-left text-muted">
        <tr>
          <th className="py-1">Method</th>
          <th>ECE</th>
          <th>MCE</th>
          <th>Brier</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.method} className="border-t border-white/10">
            <td className="py-1.5">{r.method}{r.method === data.best_method ? " (served)" : ""}</td>
            <td>{r.ece.toFixed(4)}</td>
            <td>{r.mce.toFixed(4)}</td>
            <td>{r.brier_score.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReliabilityDiagram({ data }: { data: CalibrationResponse }) {
  const bins = data.reliability_diagram_bins.filter((b) => (b.sample_count ?? 0) > 0);
  return (
    <Chart
      height={380}
      data={[
        {
          type: "scatter",
          mode: "lines",
          name: "Perfect calibration",
          x: [0, 1],
          y: [0, 1],
          line: { color: "#898781", width: 1, dash: "dash" },
          hoverinfo: "skip",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: data.best_method,
          x: bins.map((b) => b.avg_confidence ?? 0),
          y: bins.map((b) => b.empirical_accuracy ?? 0),
          line: { color: SERIES[0], width: 2 },
          marker: { size: 10, color: SERIES[0], line: { color: "#1a1a19", width: 2 } },
          hovertemplate: "confidence %{x:.3f}<br>empirical %{y:.3f}<extra></extra>",
        },
      ]}
      layout={{
        title: { text: "Reliability diagram (occupied bins only)", font: { size: 14 } },
        xaxis: { title: { text: "mean predicted confidence" }, range: [0, 1] },
        yaxis: { title: { text: "empirical failure rate" }, range: [0, 1] },
      }}
    />
  );
}

export default function Calibration() {
  const query = useCalibration();
  return (
    <div>
      <PageHeader title="📉 Confidence Calibration" subtitle="Does predicted confidence match empirical failure rate?" />
      <QueryGate query={query}>
        {(data) => (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
            <div className="lg:col-span-2 space-y-5">
              <Card title="Calibration metrics (held-out test split)">
                <MetricsTable data={data} />
              </Card>
              <Card title="Selection">
                <p className="text-sm text-ink-secondary">
                  Served model: <strong className="text-ink">{data.best_method}</strong>
                  {data.temperature !== null && <> · fitted T = {data.temperature}</>}
                </p>
                {data.selection_reason && <p className="mt-2 text-sm text-muted">{data.selection_reason}</p>}
                {data.selection_basis && (
                  <p className="mt-1 text-xs text-muted">
                    decided on {data.selection_basis} evidence, resampling by {data.resampling_unit ?? "unknown"}
                  </p>
                )}
              </Card>
            </div>
            <div className="lg:col-span-3">
              <Card>
                <ReliabilityDiagram data={data} />
              </Card>
            </div>
          </div>
        )}
      </QueryGate>
    </div>
  );
}
