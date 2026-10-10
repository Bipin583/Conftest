import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", label: "Overview", icon: "🛡️", end: true },
  { to: "/live", label: "Live PR Evaluation", icon: "🚀" },
  { to: "/calibration", label: "Confidence Calibration", icon: "📉" },
  { to: "/uncertainty", label: "Uncertainty Drilldown", icon: "🔮" },
  { to: "/baseline", label: "Baseline Comparison", icon: "📊" },
  { to: "/shap", label: "SHAP Explainability", icon: "🔍" },
  { to: "/analytics", label: "Analytics & Telemetry", icon: "📈" },
];

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 shrink-0 border-r border-white/10 p-4 md:block">
        <div className="mb-6 px-2">
          <div className="text-lg font-bold">
            <span className="gradient-title">🛡️ ConfTest</span>
          </div>
          <div className="text-xs text-muted">Confidence-Calibrated RTS</div>
        </div>
        <nav className="space-y-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? "bg-white/10 text-ink" : "text-ink-secondary hover:bg-white/5"
                }`
              }
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="flex-1">
        {/* Mobile top nav */}
        <div className="flex gap-1 overflow-x-auto border-b border-white/10 p-2 md:hidden">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-lg px-3 py-1.5 text-sm ${
                  isActive ? "bg-white/10 text-ink" : "text-ink-secondary"
                }`
              }
            >
              {item.icon} {item.label}
            </NavLink>
          ))}
        </div>
        <main className="mx-auto max-w-6xl p-4 md:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
