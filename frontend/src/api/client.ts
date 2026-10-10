import { useQuery, useMutation } from "@tanstack/react-query";
import type {
  AnalyticsResponse,
  BaselineResponse,
  CalibrationResponse,
  ExplanationsResponse,
  HeadlineResponse,
  SelectRequest,
  SelectResponse,
  UncertaintyResponse,
} from "./types";

// A failed response carries the API's `detail` string. For the 503
// missing-artifact contract that detail names the script that produces the
// report, which the pages surface verbatim.
export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.name = "ApiError";
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const useHeadline = () =>
  useQuery({ queryKey: ["headline"], queryFn: () => getJson<HeadlineResponse>("/api/v1/headline") });

export const useBaseline = () =>
  useQuery({ queryKey: ["baseline"], queryFn: () => getJson<BaselineResponse>("/api/v1/baseline") });

export const useCalibration = () =>
  useQuery({ queryKey: ["calibration"], queryFn: () => getJson<CalibrationResponse>("/api/v1/calibration") });

export const useUncertainty = () =>
  useQuery({ queryKey: ["uncertainty"], queryFn: () => getJson<UncertaintyResponse>("/api/v1/uncertainty") });

export const useExplanations = () =>
  useQuery({ queryKey: ["explanations"], queryFn: () => getJson<ExplanationsResponse>("/api/v1/explanations") });

// Persisted telemetry: every /select decision (API or GitHub webhook) is stored
// and aggregated here. Short staleTime so the page reflects new runs quickly.
export const useAnalytics = () =>
  useQuery({
    queryKey: ["analytics"],
    queryFn: () => getJson<AnalyticsResponse>("/api/v1/analytics"),
    staleTime: 5_000,
  });

export const useSelectMutation = () =>
  useMutation({
    mutationFn: (req: SelectRequest) => postJson<SelectResponse>("/api/v1/select", req),
  });
