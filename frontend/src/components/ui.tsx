import type { ReactNode } from "react";
import type { Provenance } from "../api/types";

export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-bold gradient-title">{title}</h1>
      {subtitle && <p className="mt-1 text-ink-secondary">{subtitle}</p>}
    </header>
  );
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="glass p-5">
      {title && <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">{title}</h2>}
      {children}
    </section>
  );
}

export function MetricCard({
  label,
  value,
  note,
  source,
}: {
  label: string;
  value: string | null;
  note?: string;
  source?: string;
}) {
  return (
    <div className="glass p-4">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${value ? "text-ink" : "text-warning"}`}>
        {value ?? "not measured"}
      </div>
      {note && <div className="mt-1 text-xs text-ink-secondary">{note}</div>}
      {source && <div className="mt-1 text-[11px] text-muted">source: {source}</div>}
    </div>
  );
}

// Provenance travels with every number: real measured labels vs sampled splits.
export function ProvenanceBanner({ provenance }: { provenance: Provenance }) {
  const ok = provenance.real_labels;
  return (
    <div
      className={`mb-5 rounded-lg border p-3 text-sm ${
        ok ? "border-good/40 text-good" : "border-warning/40 text-warning"
      }`}
    >
      {ok ? "✅ " : "⚠️ Labels not measured. "}
      <span className={ok ? "text-ink-secondary" : "text-ink-secondary"}>{provenance.detail}</span>
    </div>
  );
}
