// plotly.js-dist-min ships no types; we only pass it to react-plotly.js's
// factory, so an opaque module declaration is enough.
declare module "plotly.js-dist-min";

// The factory entrypoint of react-plotly.js is untyped in @types/react-plotly.js.
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  import type { PlotParams } from "react-plotly.js";
  const createPlotlyComponent: (plotly: object) => ComponentType<PlotParams>;
  export default createPlotlyComponent;
}
