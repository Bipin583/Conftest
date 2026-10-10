import { useHeadline, useBaseline } from "../api/client";
import { PageHeader, Card, MetricCard, ProvenanceBanner } from "../components/ui";
import { QueryGate } from "../components/states";
import Chart from "../components/Chart";
import { SERIES } from "../lib/plot";
import type { BaselineRow } from "../api/types";

function BaselineScatter({ rows }: { rows: BaselineRow[] }) {
  const pts = rows.filter((r) => r.time_reduction_pct !== null && r.failure_recall_pct !== null);
  const isConf = (s: string | null) => (s ?? "").includes("ConfTest");
  return (
    <Chart
      height={380}
      data={[
        {
          type: "scatter",
          mode: "markers+text",
          x: pts.map((r) => r.time_reduction_pct),
          y: pts.map((r) => r.failure_recall_pct),
          // Direct-label only ConfTest (short, to avoid overflow at x=0); the
          // rest cluster and would collide, so their identity lives in the hover.
          text: pts.map((r) => (isConf(r.strategy) ? "ConfTest" : "")),
          textposition: "middle right",
          textfont: { color: "#3987e5", size: 13 },
          customdata: pts.map((r) => r.strategy ?? ""),
          marker: {
            size: pts.map((r) => (isConf(r.strategy) ? 18 : 11)),
            color: pts.map((r) => (isConf(r.strategy) ? SERIES[0] : "#898781")),
            line: { color: "#1a1a19", width: 2 },
          },
          hovertemplate: "%{customdata}<br>Time saved %{x:.1f}%<br>Recall %{y:.1f}%<extra></extra>",
        },
      ]}
      layout={{
        title: { text: "Strategy trade-off: regression safety vs compute saved", font: { size: 14 } },
        xaxis: { title: { text: "Test execution reduction (%)" } },
        yaxis: { title: { text: "Bug-detection recall (%)" } },
        shapes: [
          {
            type: "line",
            x0: 0,
            x1: 1,
            xref: "paper",
            y0: 100,
            y1: 100,
            line: { color: "#0ca30c", width: 1, dash: "dash" },
          },
        ],
        annotations: [
          { x: 0, xref: "paper", y: 100, yanchor: "bottom", text: "100% recall", showarrow: false, font: { color: "#0ca30c", size: 11 } },
        ],
      }}
    />
  );
}

export default function Home() {
  const headline = useHeadline();
  const baseline = useBaseline();

  return (
    <div>
      <PageHeader
        title="🛡️ ConfTest: Confidence-Calibrated RTS Portal"
        subtitle="Selective prediction & uncertainty-aware regression test selection for CI/CD"
      />

      <QueryGate query={headline}>
        {(data) => (
          <>
            <ProvenanceBanner provenance={data.provenance} />
            <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {data.metrics.map((m) => (
                <MetricCard key={m.label} label={m.label} value={m.value} note={m.note} source={m.source} />
              ))}
            </div>
          </>
        )}
      </QueryGate>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <Card title="RTS Baseline Comparison (recall vs time saved)">
            <QueryGate query={baseline}>{(data) => <BaselineScatter rows={data.rows} />}</QueryGate>
          </Card>
        </div>
        <div className="lg:col-span-2">
          <Card title="System pillars">
            <ol className="list-decimal space-y-3 pl-5 text-sm text-ink-secondary">
              <li><strong className="text-ink">32-feature pipeline</strong>: churn/diff, AST, dependency-graph reachability, historical failure telemetry (anti-leakage).</li>
              <li><strong className="text-ink">5-seed deep ensemble</strong>: quantifies epistemic uncertainty as member disagreement.</li>
              <li><strong className="text-ink">Post-hoc calibration</strong>: reported only when a paired interval supports it.</li>
              <li><strong className="text-ink">Dual-mode selective policy</strong>: <code>FAST_SELECTED</code> when confident, <code>SAFE_FULL_SUITE</code> on high uncertainty or OOD diffs.</li>
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}
