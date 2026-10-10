import type { ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { ApiError } from "../api/client";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 p-8 text-ink-secondary">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-series-1 border-t-transparent" />
      {label}
    </div>
  );
}

// A missing artifact is a 503 whose detail names the producing script. Rendered
// as the blocker it is — the same contract the Streamlit `stop_on_missing_artifact`
// enforced: this page reports measurements and has none to show.
export function ArtifactBlocker({ error }: { error: unknown }) {
  const is503 = error instanceof ApiError && error.status === 503;
  const detail = error instanceof Error ? error.message : String(error);
  return (
    <div className="glass border-warning/30 p-6">
      <h3 className="mb-2 text-lg font-semibold text-warning">
        {is503 ? "Report not produced yet" : "Could not load data"}
      </h3>
      <p className="text-ink-secondary">{detail}</p>
      {is503 && (
        <p className="mt-3 text-sm text-muted">
          This page reports measurements and has none to show until the artifact is produced.
        </p>
      )}
    </div>
  );
}

// Gate a page section on a query: spinner while loading, blocker on error,
// children with the resolved data otherwise.
export function QueryGate<T>({
  query,
  children,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
}) {
  if (query.isLoading) return <Loading />;
  if (query.isError || query.data === undefined) return <ArtifactBlocker error={query.error} />;
  return <>{children(query.data)}</>;
}
