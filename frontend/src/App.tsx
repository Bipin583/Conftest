import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import LivePrEvaluation from "./pages/LivePrEvaluation";
import Calibration from "./pages/Calibration";
import Uncertainty from "./pages/Uncertainty";
import BaselineComparison from "./pages/BaselineComparison";
import ShapExplainability from "./pages/ShapExplainability";
import Analytics from "./pages/Analytics";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="live" element={<LivePrEvaluation />} />
        <Route path="calibration" element={<Calibration />} />
        <Route path="uncertainty" element={<Uncertainty />} />
        <Route path="baseline" element={<BaselineComparison />} />
        <Route path="shap" element={<ShapExplainability />} />
        <Route path="analytics" element={<Analytics />} />
      </Route>
    </Routes>
  );
}
