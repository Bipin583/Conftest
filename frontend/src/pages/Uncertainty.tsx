import { useUncertainty } from "../api/client";
import { PageHeader, Card, MetricCard } from "../components/ui";
import { QueryGate } from "../components/states";
import Chart from "../components/Chart";
import { SERIES } from "../lib/plot";

interface CurvePoint {
  coverage_pct: string;
  retained_samples: number;
  mean_prediction_error: number;
  max_retained_uncertainty: number;
}

const pctToNum = (s: string) => Number(String(s).replace("%", ""));

export default function Uncertainty() {
  const query = useUncertainty();
  return (
    <div>
      <PageHeader title="🔮 Uncertainty Drilldown" subtitle="Ensemble disagreement and the risk-coverage trade-off." />
      <QueryGate query={query}>
        {(data) => {
          const analysis = data.analysis as Record<string, number>;
          const policy = data.policy as Record<string, number>;
          const ensemble = data.ensemble as Record<string, unknown>;
          const seeds = (ensemble.seeds as number[]) ?? [];
          const numMembers = (ensemble.num_members as number) ?? seeds.length;
          const tau = policy.tau_abstain ?? null;

          const curve = ((analysis.risk_coverage_curve as unknown as CurvePoint[]) ?? [])
            .map((p) => ({ ...p, coverage: pctToNum(p.coverage_pct) }))
            .sort((a, b) => a.coverage - b.coverage);

          const at100 = curve.find((p) => p.coverage === 100)?.mean_prediction_error;
          const at90 = curve.find((p) => p.coverage === 90)?.mean_prediction_error;
          const change = at100 && at90 ? ((at90 - at100) / at100) * 100 : null;

          return (
            <>
              <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard label="Ensemble members" value={`${numMembers} models`} note={`seeds ${seeds.join(", ")}`} />
                <MetricCard label="Abstention threshold τ" value={tau === null ? null : tau.toFixed(4)} />
                <MetricCard
                  label="Error change @ 90% coverage"
                  value={change === null ? null : `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`}
                  note={at100 !== undefined ? `${at100.toFixed(4)} error at full coverage` : undefined}
                />
                <MetricCard
                  label="Uncertainty / error corr."
                  value={analysis.uncertainty_error_correlation?.toFixed(4) ?? null}
                  note={`${analysis.num_samples?.toLocaleString?.() ?? analysis.num_samples} samples`}
                />
              </div>

              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <Card title="Risk-coverage curve">
                  <Chart
                    data={[
                      {
                        type: "scatter",
                        mode: "lines+markers",
                        x: curve.map((p) => p.coverage),
                        y: curve.map((p) => p.mean_prediction_error),
                        line: { color: SERIES[0], width: 2 },
                        marker: { size: 9, color: SERIES[0], line: { color: "#1a1a19", width: 2 } },
                        hovertemplate: "coverage %{x:.0f}%<br>mean error %{y:.4f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      xaxis: { title: { text: "coverage (%)" } },
                      yaxis: { title: { text: "mean prediction error" } },
                    }}
                  />
                </Card>
                <Card title="Max retained uncertainty vs coverage">
                  <Chart
                    data={[
                      {
                        type: "scatter",
                        mode: "lines+markers",
                        x: curve.map((p) => p.coverage),
                        y: curve.map((p) => p.max_retained_uncertainty),
                        line: { color: SERIES[1], width: 2 },
                        marker: { size: 9, color: SERIES[1], line: { color: "#1a1a19", width: 2 } },
                        hovertemplate: "coverage %{x:.0f}%<br>max uncertainty %{y:.4f}<extra></extra>",
                      },
                    ]}
                    layout={{
                      xaxis: { title: { text: "coverage (%)" } },
                      yaxis: { title: { text: "max retained epistemic uncertainty" } },
                      shapes:
                        tau === null
                          ? []
                          : [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: tau, y1: tau, line: { color: "#fab219", width: 1, dash: "dash" } }],
                      annotations:
                        tau === null
                          ? []
                          : [{ x: 0, xref: "paper", y: tau, yanchor: "bottom", text: `τ = ${tau.toFixed(4)}`, showarrow: false, font: { color: "#fab219", size: 11 } }],
                    }}
                  />
                </Card>
              </div>
            </>
          );
        }}
      </QueryGate>
    </div>
  );
}
