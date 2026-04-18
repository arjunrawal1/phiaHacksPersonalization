"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type JobSummary = {
  id: string;
  status: string;
  photo_count: number;
  error: string | null;
  created_at: string;
  updated_at: string;
};

type BestMatch = {
  title: string;
  source: string;
  price: string | null;
  link: string;
  thumbnail: string;
  confidence: number;
  reasoning: string;
  source_tier: "exact" | "similar";
};

type JobItem = {
  id: string;
  photo_id: string;
  category: string;
  description: string;
  colors: string[];
  pattern: string;
  style: string;
  brand_visible: string | null;
  visibility: string;
  confidence: number;
  crop_url: string | null;
  tier: "exact" | "similar" | "generic" | "pending" | string;
  exact_matches: Array<Record<string, unknown>>;
  similar_products: Array<Record<string, unknown>>;
  best_match: BestMatch | null;
  best_match_confidence: number;
};

type JobDetail = JobSummary & {
  selected_cluster_id: string | null;
  photos: {
    id: string;
    url: string;
    width: number | null;
    height: number | null;
  }[];
  clusters: {
    id: string;
    rep_photo_id: string;
    rep_bbox: { left: number; top: number; width: number; height: number };
    member_count: number;
    source_url: string;
  }[];
  items: JobItem[];
};

type Props = {
  backendUrl: string;
};

const POLL_MS = 1500;

export function PipelineDashboard({ backendUrl }: Props) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshingItemId, setRefreshingItemId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const jobsRes = await fetch(`${backendUrl}/api/jobs`, { cache: "no-store" });
      if (!jobsRes.ok) throw new Error(`Jobs HTTP ${jobsRes.status}`);
      const nextJobs = (await jobsRes.json()) as JobSummary[];
      setJobs(nextJobs);

      const targetId = selectedJobId ?? nextJobs[0]?.id ?? null;
      if (!targetId) {
        setDetail(null);
        setSelectedJobId(null);
        return;
      }
      if (targetId !== selectedJobId) setSelectedJobId(targetId);

      const detailRes = await fetch(`${backendUrl}/api/jobs/${targetId}`, { cache: "no-store" });
      if (!detailRes.ok) throw new Error(`Detail HTTP ${detailRes.status}`);
      const nextDetail = (await detailRes.json()) as JobDetail;
      setDetail(nextDetail);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [backendUrl, selectedJobId]);

  useEffect(() => {
    const kickoff = setTimeout(() => {
      void load();
    }, 0);
    const id = setInterval(() => void load(), POLL_MS);
    return () => {
      clearTimeout(kickoff);
      clearInterval(id);
    };
  }, [load]);

  const grouped = useMemo(() => {
    if (!detail) return [] as { photo: JobDetail["photos"][number]; items: JobItem[] }[];
    const byPhoto = new Map<string, JobItem[]>();
    for (const item of detail.items) {
      const arr = byPhoto.get(item.photo_id) ?? [];
      arr.push(item);
      byPhoto.set(item.photo_id, arr);
    }
    return detail.photos
      .map((photo) => ({ photo, items: byPhoto.get(photo.id) ?? [] }))
      .filter((section) => section.items.length > 0);
  }, [detail]);

  const counts = useMemo(() => {
    if (!detail) return { exact: 0, similar: 0, pending: 0, generic: 0 };
    return {
      exact: detail.items.filter((i) => i.tier === "exact").length,
      similar: detail.items.filter((i) => i.tier === "similar").length,
      pending: detail.items.filter((i) => i.tier === "pending").length,
      generic: detail.items.filter((i) => i.tier === "generic").length,
    };
  }, [detail]);

  async function refreshItem(itemId: string) {
    if (refreshingItemId) return;
    setRefreshingItemId(itemId);
    try {
      const res = await fetch(`${backendUrl}/api/items/${itemId}/refresh`, {
        method: "POST",
      });
      if (!res.ok) {
        throw new Error(`Refresh HTTP ${res.status}`);
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshingItemId(null);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <section className="rounded-xl border border-border bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Jobs</h2>
          <span className="text-xs text-muted-foreground">{jobs.length}</span>
        </div>

        <div className="space-y-2">
          {jobs.length === 0 ? (
            <div className="rounded-md border border-border bg-muted px-3 py-4 text-sm text-muted-foreground">
              No jobs yet. Start a mobile sync.
            </div>
          ) : (
            jobs.map((job) => (
              <button
                key={job.id}
                type="button"
                onClick={() => setSelectedJobId(job.id)}
                className={`w-full rounded-md border px-3 py-3 text-left transition-colors ${
                  selectedJobId === job.id
                    ? "border-accent bg-accent/10"
                    : "border-border bg-card hover:bg-muted"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="font-mono text-xs text-muted-foreground">{job.id.slice(0, 8)}</span>
                  <span className="text-xs font-medium uppercase tracking-wide text-foreground">{job.status}</span>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">{job.photo_count} photos</div>
              </button>
            ))
          )}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card p-5">
        {!detail ? (
          <div className="text-sm text-muted-foreground">Waiting for pipeline data...</div>
        ) : (
          <>
            <header className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-xl font-semibold tracking-tight">Job {detail.id.slice(0, 8)}</h2>
                <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs font-semibold uppercase tracking-wide">
                  {detail.status}
                </span>
              </div>
              <p className="text-sm text-muted-foreground">
                {detail.photo_count} photos • {grouped.length} outfit photos • {detail.items.length} items
              </p>
              <p className="text-sm text-muted-foreground">
                {counts.exact} exact • {counts.similar} similar • {counts.pending} pending • {counts.generic} no-match
              </p>
              {detail.error ? <p className="text-sm text-destructive">{detail.error}</p> : null}
              {error ? <p className="text-sm text-destructive">{error}</p> : null}
            </header>

            {detail.clusters.length > 0 ? (
              <section className="mt-6">
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  Face Clusters
                </h3>
                <div className="flex flex-wrap gap-3">
                  {detail.clusters.slice(0, 10).map((cluster) => (
                    <div
                      key={cluster.id}
                      className={`w-28 overflow-hidden rounded-lg border ${
                        detail.selected_cluster_id === cluster.id ? "border-accent" : "border-border"
                      }`}
                    >
                      <img src={cluster.source_url} alt="face cluster" className="h-20 w-full object-cover" />
                      <div className="p-2 text-xs">
                        <div className="font-medium">{cluster.member_count} photos</div>
                        {detail.selected_cluster_id === cluster.id ? <div className="text-accent">selected</div> : null}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            <section className="mt-6 space-y-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Labeled Photos</h3>
              {grouped.length === 0 ? (
                <div className="rounded-md border border-border bg-muted px-3 py-6 text-sm text-muted-foreground">
                  No labeled items yet.
                </div>
              ) : (
                grouped.map(({ photo, items }) => (
                  <div key={photo.id} className="grid gap-3 md:grid-cols-[220px_minmax(0,1fr)]">
                    <img src={photo.url} alt="uploaded" className="h-52 w-full rounded-lg border border-border object-cover" />
                    <div className="space-y-2">
                      {items.map((item) => (
                        <div key={item.id} className="rounded-lg border border-border bg-muted p-2">
                          <div className="flex items-start gap-3">
                            {item.crop_url ? (
                              <img src={item.crop_url} alt="crop" className="h-16 w-16 rounded-md object-cover" />
                            ) : (
                              <div className="h-16 w-16 rounded-md bg-card" />
                            )}
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className={`rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${
                                  item.tier === "exact"
                                    ? "bg-emerald-600 text-white"
                                    : item.tier === "similar"
                                    ? "bg-emerald-100 text-emerald-700"
                                    : item.tier === "pending"
                                    ? "bg-amber-100 text-amber-700"
                                    : "bg-slate-200 text-slate-700"
                                }`}>
                                  {item.tier}
                                </span>
                                <span className="text-xs text-muted-foreground">{item.category}</span>
                                {item.brand_visible ? <span className="text-xs font-semibold">{item.brand_visible}</span> : null}
                                <button
                                  type="button"
                                  onClick={() => refreshItem(item.id)}
                                  disabled={refreshingItemId !== null}
                                  className="ml-auto rounded border border-border bg-card px-2 py-1 text-[11px] hover:bg-background disabled:opacity-50"
                                >
                                  {refreshingItemId === item.id ? "Refreshing..." : "Refresh"}
                                </button>
                              </div>

                              <p className="mt-1 truncate text-sm font-semibold">{item.description}</p>
                              <p className="text-xs text-muted-foreground">
                                {Math.round(item.confidence * 100)}% extraction confidence
                              </p>
                            </div>
                          </div>

                          {item.best_match ? (
                            <a
                              href={item.best_match.link || "#"}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="mt-2 flex items-center gap-2 rounded-md border border-border bg-card p-2"
                            >
                              {item.best_match.thumbnail ? (
                                <img src={item.best_match.thumbnail} alt="candidate" className="h-14 w-14 rounded object-cover" />
                              ) : (
                                <div className="h-14 w-14 rounded bg-muted" />
                              )}
                              <div className="min-w-0 flex-1">
                                <p className="truncate text-xs font-semibold">{item.best_match.title}</p>
                                <p className="text-[11px] text-muted-foreground">
                                  {item.best_match.price ?? ""} {item.best_match.source ? `• ${item.best_match.source}` : ""}
                                </p>
                                <p className="text-[11px] text-muted-foreground">
                                  match {Math.round((item.best_match_confidence || 0) * 100)}%
                                </p>
                              </div>
                            </a>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </section>
          </>
        )}
      </section>
    </div>
  );
}
