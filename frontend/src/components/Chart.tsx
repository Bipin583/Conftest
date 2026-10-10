import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";
import { baseConfig, baseLayout } from "../lib/plot";

// Build react-plotly against the minified dist bundle rather than the default
// full build, keeping the vendor chunk smaller.
const Plot = createPlotlyComponent(Plotly as object);

export default function Chart({
  data,
  layout,
  height = 360,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
}) {
  return (
    <Plot
      data={data}
      layout={{ ...baseLayout(), height, ...layout }}
      config={baseConfig}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
