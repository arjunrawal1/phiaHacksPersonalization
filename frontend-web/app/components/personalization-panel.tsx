"use client";

import { useCallback, useEffect, useState } from "react";

type Props = {
  backendUrl: string;
};

type JobSummary = {
  id: string;
};

type PhotoInsight = {
  photo_id: string;
  summary: string;
  style_tags: string[];
  brand_hints: string[];
  captured_at?: string | null;
  latitude?: number | null;
  longitude?: number | null;
};

type PersonalizationSummary = {
  job_id: string;
  generated_at: string;
  source: "openai" | "fallback" | "cached";
  photo_count: number;
  collection_titles: string[];
  notifications: string[];
  favorite_brands: string[];
  style_summary: string;
  photo_insights: PhotoInsight[];
};

const POLL_MS = 8000;

function formatWhen(raw: string | null | undefined): string {
  if (!raw) return "time unknown";
  const dt = new Date(raw);
  if (Number.isNaN(dt.getTime())) return "time unknown";
  return dt.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatLocation(lat?: number | null, lon?: number | null): string {
  if (typeof lat !== "number" || typeof lon !== "number") return "location unavailable";
  return `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
}

export function PersonalizationPanel({ backendUrl }: Props) {
  const [summary, setSummary] = useState<PersonalizationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const jobsRes = await fetch(`${backendUrl}/api/jobs`, { cache: "no-store" });
      if (!jobsRes.ok) {
        setError(`Failed to fetch jobs (${jobsRes.status})`);
        setLoading(false);
        return;
      }
      const jobs = (await jobsRes.json()) as JobSummary[];
      const latestJobId = jobs[0]?.id;
      if (!latestJobId) {
        setSummary(null);
        setError(null);
        setLoading(false);
        return;
      }

      const res = await fetch(
        `${backendUrl}/api/personalization?job_id=${encodeURIComponent(latestJobId)}`,
        { cache: "no-store" }
      );
      if (!res.ok) {
        setError(`Failed to fetch personalization (${res.status})`);
        setLoading(false);
        return;
      }
      const payload = (await res.json()) as PersonalizationSummary;
      setSummary(payload);
      setError(null);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }, [backendUrl]);

  useEffect(() => {
    const kickoff = setTimeout(() => void load(), 0);
    const id = setInterval(() => void load(), POLL_MS);
    return () => {
      clearTimeout(kickoff);
      clearInterval(id);
    };
  }, [load]);

  return (
    <section className="phia-panel p-4 sm:p-6">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Personalization
        </h2>
        {summary ? (
          <span className="rounded-full border border-border bg-background px-2 py-0.5 text-[11px] text-muted-foreground">
            {summary.source} • {summary.photo_count} photos
          </span>
        ) : null}
      </div>

      {loading ? (
        <div className="rounded-xl border border-dashed border-border bg-background px-3 py-4 text-sm text-muted-foreground">
          Building personalized style intelligence...
        </div>
      ) : null}

      {!loading && error ? (
        <div className="rounded-xl border border-dashed border-border bg-background px-3 py-4 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      {!loading && !error && !summary ? (
        <div className="rounded-xl border border-dashed border-border bg-background px-3 py-4 text-sm text-muted-foreground">
          Sync photos to generate personalization insights.
        </div>
      ) : null}

      {!loading && !error && summary ? (
        <div className="space-y-4">
          <div className="rounded-2xl border border-border bg-background p-4">
            <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Style Summary
            </div>
            <p className="mt-2 text-sm leading-6 text-foreground">{summary.style_summary || "No summary yet."}</p>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-2xl border border-border bg-background p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Collection Titles
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {(summary.collection_titles || []).map((title, idx) => (
                  <span
                    key={`collection-title-${idx}-${title}`}
                    className="rounded-full border border-border bg-muted/30 px-2 py-1 text-[11px] text-foreground"
                  >
                    {title}
                  </span>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-background p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Favorite Brands
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {(summary.favorite_brands || []).map((brand, idx) => (
                  <span
                    key={`favorite-brand-${idx}-${brand}`}
                    className="rounded-full border border-border bg-muted/30 px-2 py-1 text-[11px] text-foreground"
                  >
                    {brand}
                  </span>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-background p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Notifications
              </div>
              <div className="mt-2 space-y-1.5">
                {(summary.notifications || []).map((note, idx) => (
                  <div
                    key={`notification-${idx}-${note}`}
                    className="rounded-lg border border-border bg-muted/30 px-2 py-1.5 text-[11px] text-foreground"
                  >
                    {note}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-background p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Photo Insights
            </div>
            <div className="phia-scroll mt-2 flex gap-2 overflow-x-auto pb-1">
              {(summary.photo_insights || []).map((insight) => (
                <div
                  key={`photo-insight-${insight.photo_id}`}
                  className="w-72 shrink-0 rounded-xl border border-border bg-card p-2.5"
                >
                  <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                    {formatWhen(insight.captured_at)} • {formatLocation(insight.latitude, insight.longitude)}
                  </div>
                  <div className="mt-1 text-xs font-semibold text-foreground">{insight.summary}</div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {insight.style_tags.map((tag) => (
                      <span
                        key={`${insight.photo_id}-tag-${tag}`}
                        className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[10px] text-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
