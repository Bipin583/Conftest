import { useExplanations } from "../api/client";
import { PageHeader, Card } from "../components/ui";
import { QueryGate } from "../components/states";
import Chart from "../components/Chart";
import { SERIES } from "../lib/plot";

interface Importance {
  feature: string;
  mean_abs_shap: number;
}

// Group a feature into one of the pipeline's four families by name prefix.
function family(feature: string): string {
  if (feature.startsWith("dep_")) return "Dependency graph";
  if (feature.startsWith("ast_") || feature.includes("cyclo")) return "AST / complexity";
  if (feature.startsWith("hist_") || feature.includes("fail")) return "Failure history";
  if (feature.startsWith("churn") || feature.includes("diff") || feature.includes("line") || feature.includes("changed") || feature.includes("mod_count"))
    return "Churn / diff";
  return "Other";
}

export default function ShapExplainability() {
  const query = useExplanations();
  return (
    <div>
      <PageHeader title="🔍 SHAP Explainability" subtitle="Which features drive the model's failure-risk scores." />
      <QueryGate query={query}>
        {(data) => {
          const importance = ([...(data.global_shap_importance ?? [])] as Importance[]).sort(
            (a, b) => b.mean_abs_shap - a.mean_abs_shap,
          );
          const top = importance.slice(0, 15).reverse();

          const byFamily = new Map<string, number>();
          for (const row of importance) {
            byFamily.set(family(row.feature), (byFamily.get(family(row.feature)) ?? 0) + row.mean_abs_shap);
          }
          const families = [...byFamily.entries()];

          return (
            <>
              {data.labels_measured === false && (
                <div className="mb-5 rounded-lg border border-warning/40 p-3 text-sm text-warning">
                  ⚠️ SHAP computed on sampled-label data, not measured test executions.
                </div>
              )}
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
                <div className="lg:col-span-3">
                  <Card title="Global feature importance (mean |SHAP|)">
                    <Chart
                      height={Math.max(260, top.length * 26)}
                      data={[
                        {
                          type: "bar",
                          orientation: "h",
                          y: top.map((r) => r.feature),
                          x: top.map((r) => r.mean_abs_shap),
                          marker: { color: SERIES[0], line: { color: "#1a1a19", width: 1 } },
                          hovertemplate: "%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>",
                        },
                      ]}
                      layout={{ xaxis: { title: { text: "mean |SHAP|" } }, yaxis: { automargin: true, tickfont: { color: "#898781" } }, margin: { l: 20, r: 20, t: 20, b: 40 } }}
                    />
                  </Card>
                </div>
                <div className="lg:col-span-2">
                  <Card title="Importance by feature family">
                    <Chart
                      height={320}
                      data={[
                        {
                          type: "pie",
                          labels: families.map((f) => f[0]),
                          values: families.map((f) => f[1]),
                          marker: { colors: SERIES, line: { color: "#1a1a19", width: 2 } },
                          textfont: { color: "#f0efec" },
                          hovertemplate: "%{label}<br>%{percent}<extra></extra>",
                        },
                      ]}
                      layout={{ showlegend: true }}
                    />
                  </Card>
                </div>
              </div>
            </>
          );
        }}
      </QueryGate>
    </div>
  );
}
