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

type PipelineEvent = {
  at: string;
  type: string;
  message: string;
  data?: Record<string, unknown>;
};

type JobDebug = {
  stages?: Record<string, string>;
  upload?: {
    expected_count?: number;
    saved_count?: number;
    started_at?: string;
    finished_at?: string;
    user_name?: string | null;
  };
  face_analysis?: {
    started_at?: string;
    user_name?: string | null;
    total_clusters?: number;
    total_detected_faces?: number;
    inserted_cluster_count?: number;
    cluster_member_counts?: number[];
    faces_by_photo?: Record<string, number>;
  };
  serp_lookup?: {
    query?: string;
    candidate_count?: number;
    candidate_urls?: string[];
    downloaded_count?: number;
    downloaded_urls?: string[];
    filtered_face_count?: number;
    filtered_face_urls?: string[];
  };
  auto_face_score?: {
    state?: "running" | "complete";
    mode?: string;
    reason?: string | null;
    auto_selected?: boolean;
    selected_cluster_id?: string | null;
    top_cluster_id?: string | null;
    top_score?: number;
    top_alignment?: number;
    top_frequency?: number;
    second_score?: number;
    margin?: number;
    top_matched_urls?: string[];
    thresholds?: {
      score?: number;
      margin?: number;
      alignment_floor?: number;
    };
    scored_clusters?: Array<{
      cluster_id: string;
      member_count?: number;
      frequency: number;
      alignment: number;
      score: number;
      matched_urls?: string[];
    }>;
  };
  clothing_extraction?: {
    cluster_id?: string;
    started_at?: string;
    finished_at?: string;
    item_count?: number;
    photo_count?: number;
  };
  events?: PipelineEvent[];
};

type FaceDetection = {
  id: string;
  photo_id: string;
  cluster_id: string | null;
  bbox: { left: number; top: number; width: number; height: number };
  confidence: number;
};

type ClothingItem = {
  id: string;
  photo_id: string;
  category: string;
  description: string;
  confidence: number;
  tier: string;
  bounding_box: {
    x?: number;
    y?: number;
    w?: number;
    h?: number;
    left?: number;
    top?: number;
    width?: number;
    height?: number;
  };
  crop_url?: string | null;
  best_match?: {
    title?: string;
    source?: string;
    price?: string | null;
    link?: string;
    thumbnail?: string;
  } | null;
  best_match_confidence?: number;
};

type JobDetail = JobSummary & {
  selected_cluster_id: string | null;
  photos: {
    id: string;
    url: string;
    width: number | null;
    height: number | null;
  }[];
  face_detections: FaceDetection[];
  clusters: {
    id: string;
    rep_photo_id: string;
    rep_bbox: { left: number; top: number; width: number; height: number };
    rep_aspect_ratio?: number;
    member_count: number;
    source_url: string;
  }[];
  items: ClothingItem[];
  debug?: JobDebug;
};

type Props = {
  backendUrl: string;
};

const POLL_MS = 1200;

function isSerpProxyUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    return u.hostname.includes("serpapi.com") && u.pathname.includes("/searches/");
  } catch {
    return raw.includes("serpapi.com/searches/");
  }
}

function imageKey(raw: string): string {
  try {
    const u = new URL(raw);
    const normalizedPath = decodeURIComponent(u.pathname).replace(/\/+$/, "").toLowerCase();
    return `${u.hostname.toLowerCase()}${normalizedPath}`;
  } catch {
    return raw.trim().toLowerCase();
  }
}

function dedupeImageUrls(urls: string[]): string[] {
  const nonProxy = urls.filter((u) => !isSerpProxyUrl(u));
  const source = nonProxy.length > 0 ? nonProxy : urls;

  const out: string[] = [];
  const seen = new Set<string>();
  for (const u of source) {
    const key = imageKey(u);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(u);
  }
  return out;
}

function getFaceCropStyle(
  bb: { left: number; top: number; width: number; height: number },
  aspectRatio: number,
  size: number
): React.CSSProperties {
  const safeAspectRatio = aspectRatio > 0 ? aspectRatio : 1;
  const visibleFraction = Math.max(bb.width, bb.height) * 1.8;
  const scale = size / Math.max(visibleFraction, 0.0001);

  const imgW = scale * (safeAspectRatio >= 1 ? 1 : safeAspectRatio);
  const imgH = scale * (safeAspectRatio >= 1 ? 1 / safeAspectRatio : 1);

  const centerX = bb.left + bb.width / 2;
  const centerY = bb.top + bb.height / 2;

  return {
    position: "absolute",
    width: imgW,
    height: imgH,
    left: size / 2 - centerX * imgW,
    top: size / 2 - centerY * imgH,
    maxWidth: "none",
    maxHeight: "none",
    objectFit: "fill",
  };
}

function getItemBoxStyle(
  bb: ClothingItem["bounding_box"] | null | undefined
): React.CSSProperties {
  const x = Math.max(0, Math.min(1, Number(bb?.x ?? bb?.left ?? 0)));
  const y = Math.max(0, Math.min(1, Number(bb?.y ?? bb?.top ?? 0)));
  const w = Math.max(0, Math.min(1, Number(bb?.w ?? bb?.width ?? 0)));
  const h = Math.max(0, Math.min(1, Number(bb?.h ?? bb?.height ?? 0)));

  return {
    position: "absolute",
    left: `${x * 100}%`,
    top: `${y * 100}%`,
    width: `${w * 100}%`,
    height: `${h * 100}%`,
  };
}

export function PipelineDashboard({ backendUrl }: Props) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busySelectClusterId, setBusySelectClusterId] = useState<string | null>(null);
  const [visibleSerpUrls, setVisibleSerpUrls] = useState<string[]>([]);

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
        setVisibleSerpUrls([]);
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
    const kickoff = setTimeout(() => void load(), 0);
    const id = setInterval(() => void load(), POLL_MS);
    return () => {
      clearTimeout(kickoff);
      clearInterval(id);
    };
  }, [load]);

  const detectionsByPhoto = useMemo(() => {
    const map = new Map<string, FaceDetection[]>();
    for (const det of detail?.face_detections ?? []) {
      const arr = map.get(det.photo_id) ?? [];
      arr.push(det);
      map.set(det.photo_id, arr);
    }
    return map;
  }, [detail?.face_detections]);

  const clusterById = useMemo(() => {
    const map = new Map<string, JobDetail["clusters"][number]>();
    for (const c of detail?.clusters ?? []) map.set(c.id, c);
    return map;
  }, [detail?.clusters]);

  const selectedCluster = detail?.selected_cluster_id ? clusterById.get(detail.selected_cluster_id) ?? null : null;
  const selectedClusterPhoto = selectedCluster
    ? detail?.photos.find((p) => p.id === selectedCluster.rep_photo_id) ?? null
    : null;

  const serpCandidates = useMemo(
    () => dedupeImageUrls(detail?.debug?.serp_lookup?.candidate_urls ?? []),
    [detail?.debug?.serp_lookup?.candidate_urls]
  );
  const personName = detail?.debug?.upload?.user_name ?? "user";
  const autoFaceState = detail?.debug?.auto_face_score?.state;
  const isAutoScoreRunning = autoFaceState === "running";
  const topScore = detail?.debug?.auto_face_score?.top_score;
  const shouldAskUserConfirm =
    detail?.status === "awaiting_face_pick" ||
    (!!detail &&
      !selectedCluster &&
      !isAutoScoreRunning &&
      detail?.debug?.auto_face_score?.auto_selected !== true);
  const selectedMatchedUrlKeySet = useMemo(() => {
    const selectedId = selectedCluster?.id;
    if (!selectedId) return new Set<string>();
    const scored = detail?.debug?.auto_face_score?.scored_clusters ?? [];
    const row = scored.find((r) => r.cluster_id === selectedId);
    const urls = row?.matched_urls ?? [];
    return new Set(urls.map((u) => imageKey(u)));
  }, [detail?.debug?.auto_face_score?.scored_clusters, selectedCluster?.id]);

  const itemsByPhoto = useMemo(() => {
    const map = new Map<string, ClothingItem[]>();
    for (const item of detail?.items ?? []) {
      const arr = map.get(item.photo_id) ?? [];
      arr.push(item);
      map.set(item.photo_id, arr);
    }
    return map;
  }, [detail?.items]);

  const selectedClusterPhotoIds = useMemo(() => {
    const ids = new Set<string>();
    if (!selectedCluster) return ids;
    for (const det of detail?.face_detections ?? []) {
      if (det.cluster_id === selectedCluster.id) {
        ids.add(det.photo_id);
      }
    }
    return ids;
  }, [detail?.face_detections, selectedCluster]);

  const chosenFacePhotoRows = useMemo(() => {
    if (!detail) return [];
    return detail.photos
      .filter((p) => selectedClusterPhotoIds.has(p.id))
      .map((photo) => ({
        photo,
        items: itemsByPhoto.get(photo.id) ?? [],
      }));
  }, [detail, selectedClusterPhotoIds, itemsByPhoto]);

  useEffect(() => {
    let cancelled = false;

    const checkUrl = (url: string): Promise<{ url: string; ok: boolean }> =>
      new Promise((resolve) => {
        const img = new Image();
        let done = false;
        const finish = (ok: boolean) => {
          if (done) return;
          done = true;
          resolve({ url, ok });
        };

        const timer = setTimeout(() => finish(false), 7000);
        img.onload = () => {
          clearTimeout(timer);
          finish(true);
        };
        img.onerror = () => {
          clearTimeout(timer);
          finish(false);
        };
        img.referrerPolicy = "no-referrer";
        img.src = url;
      });

    const run = async () => {
      if (serpCandidates.length === 0) {
        if (!cancelled) setVisibleSerpUrls([]);
        return;
      }
      const checks = await Promise.all(serpCandidates.map((url) => checkUrl(url)));
      if (cancelled) return;
      setVisibleSerpUrls(checks.filter((c) => c.ok).map((c) => c.url));
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [serpCandidates]);

  const chooseCluster = useCallback(
    async (clusterId: string) => {
      if (!detail) return;
      setBusySelectClusterId(clusterId);
      try {
        const res = await fetch(`${backendUrl}/api/jobs/${detail.id}/select-cluster`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cluster_id: clusterId }),
        });
        if (!res.ok) throw new Error(`Select HTTP ${res.status}`);
        const data = (await res.json()) as { job: JobDetail };
        setDetail(data.job);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusySelectClusterId(null);
      }
    },
    [backendUrl, detail]
  );

  return (
    <div className="space-y-5">
      <header className="rounded-xl border border-border bg-card px-5 py-4">
        <h1 className="text-xl font-semibold tracking-tight">Phia Personalization (Behind the Scenes)</h1>
      </header>

      <section className="rounded-xl border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Jobs</h2>
        <div className="flex gap-3 overflow-x-auto pb-1">
          {jobs.map((job) => (
            <button
              key={job.id}
              type="button"
              onClick={() => {
                setSelectedJobId(job.id);
              }}
              className={`min-w-[220px] rounded-lg border px-3 py-2 text-left transition-colors ${
                selectedJobId === job.id ? "border-accent bg-accent/10" : "border-border bg-muted hover:bg-card"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-muted-foreground">{job.id.slice(0, 8)}</span>
                <span className="text-[11px] font-semibold uppercase tracking-wide">{job.status}</span>
              </div>
              <div className="mt-1 text-xs text-muted-foreground">{job.photo_count} photos</div>
            </button>
          ))}
        </div>
      </section>

      {!detail ? (
        <section className="rounded-xl border border-border bg-card p-5 text-sm text-muted-foreground">
          Waiting for selected job...
        </section>
      ) : (
        <section className="space-y-5 rounded-xl border border-border bg-card p-5">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-lg font-semibold">Selected Job {detail.id.slice(0, 8)}</h2>
            <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs font-semibold uppercase tracking-wide">
              {detail.status}
            </span>
            {detail.error ? <span className="text-sm text-destructive">{detail.error}</span> : null}
            {error ? <span className="text-sm text-destructive">{error}</span> : null}
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <section className="space-y-3 xl:col-span-1">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Uploaded Photos</h3>
              <div className="space-y-3">
                {detail.photos.map((photo) => {
                  const detections = detectionsByPhoto.get(photo.id) ?? [];
                  const aspect =
                    (photo.width ?? 0) > 0 && (photo.height ?? 0) > 0
                      ? (photo.width ?? 1) / (photo.height ?? 1)
                      : 1;
                  return (
                    <div key={photo.id} className="rounded-lg border border-border bg-muted p-2">
                      <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-3">
                        <img src={photo.url} alt="uploaded" className="h-24 w-full rounded-md object-cover" />
                        <div>
                          <div className="mb-1 flex items-center justify-between">
                            <span className="font-mono text-[11px] text-muted-foreground">{photo.id.slice(0, 6)}</span>
                            <span className="text-xs font-semibold">{detections.length} faces</span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {detections.map((det) => (
                              <div
                                key={det.id}
                                className="relative h-12 w-12 overflow-hidden rounded-full border border-border bg-card"
                              >
                                <img src={photo.url} alt="face crop" style={getFaceCropStyle(det.bbox, aspect, 48)} />
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="space-y-3 xl:col-span-1">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Aggregated Faces</h3>
              <div className="grid grid-cols-3 gap-3 lg:grid-cols-4">
                {detail.clusters.map((cluster) => (
                  <button
                    key={cluster.id}
                    type="button"
                    disabled={busySelectClusterId !== null}
                    onClick={() => void chooseCluster(cluster.id)}
                    className={`rounded-lg border bg-muted p-2 text-center transition-colors ${
                      detail.selected_cluster_id === cluster.id ? "border-accent" : "border-border"
                    }`}
                  >
                    <div className="relative mx-auto h-14 w-14 overflow-hidden rounded-full border border-border bg-card">
                      <img
                        src={cluster.source_url}
                        alt="aggregated face"
                        style={getFaceCropStyle(cluster.rep_bbox, cluster.rep_aspect_ratio ?? 1, 56)}
                      />
                    </div>
                    <div className="mt-1 text-xs font-semibold">{cluster.member_count}</div>
                    <div className="text-[10px] text-muted-foreground">photos</div>
                  </button>
                ))}
              </div>
            </section>

            <section className="space-y-3 xl:col-span-1">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                {`Google Image Response For "${personName}"`}
              </h3>
              <div className="grid grid-cols-3 gap-2 lg:grid-cols-4">
                {visibleSerpUrls.slice(0, 24).map((url, idx) => (
                  <a
                    key={`${idx}-${url}`}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`overflow-hidden rounded-md border bg-muted ${
                      selectedMatchedUrlKeySet.has(imageKey(url))
                        ? "border-4 border-emerald-600"
                        : "border-border"
                    }`}
                  >
                    <img
                      src={url}
                      alt=""
                      className="h-16 w-full object-cover"
                      onError={() => setVisibleSerpUrls((prev) => prev.filter((u) => u !== url))}
                    />
                  </a>
                ))}
              </div>
            </section>
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
            <section className="rounded-lg border border-border bg-muted p-3">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                Ask User To Confirm Their Face
              </h3>
              <div className="mt-2 space-y-1 text-sm">
                <p>
                  Confidence score:{" "}
                  {isAutoScoreRunning ? (
                    <span className="font-semibold text-muted-foreground">Computing...</span>
                  ) : (
                    <span className="font-semibold">
                      {typeof topScore === "number" ? `${Math.round(topScore * 100)}%` : "N/A"}
                    </span>
                  )}
                </p>
                <p>
                  Ask user to confirm:{" "}
                  {isAutoScoreRunning ? (
                    <span className="font-semibold text-muted-foreground">Determining...</span>
                  ) : (
                    <span className={`font-semibold ${shouldAskUserConfirm ? "text-amber-700" : "text-emerald-700"}`}>
                      {shouldAskUserConfirm ? "YES" : "NO"}
                    </span>
                  )}
                </p>
              </div>
              {shouldAskUserConfirm ? (
                <p className="mt-2 text-xs text-muted-foreground">
                  Mobile app will prompt user to pick a face. Once selected, this view updates automatically.
                </p>
              ) : isAutoScoreRunning ? (
                <p className="mt-2 text-xs text-muted-foreground">
                  Comparing the highest-frequency face against online references...
                </p>
              ) : null}
            </section>

            <section className="rounded-lg border border-border bg-muted p-3 text-center">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">Chosen Face</h3>
              {selectedCluster ? (
                <>
                  <div className="relative mx-auto mt-2 h-20 w-20 overflow-hidden rounded-full border border-border bg-card">
                    <img
                      src={selectedCluster.source_url}
                      alt="chosen face"
                      style={getFaceCropStyle(selectedCluster.rep_bbox, selectedCluster.rep_aspect_ratio ?? 1, 80)}
                    />
                  </div>
                  <div className="mt-2 text-sm font-semibold">{selectedCluster.member_count} photos</div>
                  <div className="text-xs text-muted-foreground">cluster {selectedCluster.id.slice(0, 8)}</div>
                  {selectedClusterPhoto ? (
                    <div className="mt-1 text-[11px] text-muted-foreground">from photo {selectedClusterPhoto.id.slice(0, 6)}</div>
                  ) : null}
                </>
              ) : (
                <div className="mt-3 rounded-md border border-dashed border-border bg-card px-3 py-4 text-sm text-muted-foreground">
                  {shouldAskUserConfirm ? "Waiting on user response" : "No face chosen yet"}
                </div>
              )}
            </section>
          </div>

          <section className="space-y-3 rounded-lg border border-border bg-muted p-3">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
              Chosen Face Photo Breakdown
            </h3>
            {!selectedCluster ? (
              <div className="rounded-md border border-dashed border-border bg-card px-3 py-4 text-sm text-muted-foreground">
                Choose a face cluster to see photo/item/product breakdown.
              </div>
            ) : chosenFacePhotoRows.length === 0 ? (
              <div className="rounded-md border border-dashed border-border bg-card px-3 py-4 text-sm text-muted-foreground">
                No photos found for the chosen face yet.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid gap-3 lg:grid-cols-2">
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Photos With Chosen Face
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Products In This Photo
                  </div>
                </div>

                {chosenFacePhotoRows.map(({ photo, items }) => {
                  const photoAspect =
                    (photo.width ?? 0) > 0 && (photo.height ?? 0) > 0
                      ? `${photo.width} / ${photo.height}`
                      : "1 / 1";
                  return (
                    <div key={photo.id} className="grid gap-3 rounded-lg border border-border bg-card p-3 lg:grid-cols-2">
                    <section>
                      <div
                        className="relative w-full overflow-hidden rounded-md border border-border bg-background"
                        style={{ aspectRatio: photoAspect }}
                      >
                        <img src={photo.url} alt="item boxes overlay" className="absolute inset-0 h-full w-full object-fill" />
                        {items.map((item) => (
                          <div key={`${photo.id}-${item.id}`} className="pointer-events-none absolute" style={getItemBoxStyle(item.bounding_box)}>
                            <div className="h-full w-full border-2 border-sky-500" />
                            <span className="absolute left-0 top-0 bg-sky-500 px-1 py-0.5 text-[10px] font-semibold text-white">
                              {item.category}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                        <span>photo {photo.id.slice(0, 8)}</span>
                        <span>{items.length} items detected</span>
                      </div>
                    </section>

                    <section>
                      {items.length === 0 ? (
                        <div className="rounded-md border border-dashed border-border bg-background px-3 py-4 text-sm text-muted-foreground">
                          No extracted items yet.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {items.map((item) => {
                            const match = item.best_match;
                            return (
                              <div key={item.id} className="rounded-md border border-border bg-background p-2">
                                <div className="flex items-start gap-2">
                                  {item.crop_url ? (
                                    <img src={item.crop_url} alt="item crop" className="h-12 w-12 rounded object-cover" />
                                  ) : (
                                    <div className="h-12 w-12 rounded bg-muted" />
                                  )}
                                  <div className="min-w-0 flex-1">
                                    <div className="truncate text-xs font-semibold">{item.description}</div>
                                    <div className="text-[11px] text-muted-foreground">
                                      {item.category} • {item.tier}
                                    </div>
                                  </div>
                                </div>
                                {match?.link ? (
                                  <a
                                    href={match.link}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="mt-2 flex items-center gap-2 rounded border border-border bg-card p-2"
                                  >
                                    {match.thumbnail ? (
                                      <img src={match.thumbnail} alt="matched product" className="h-10 w-10 rounded object-cover" />
                                    ) : (
                                      <div className="h-10 w-10 rounded bg-muted" />
                                    )}
                                    <div className="min-w-0 flex-1">
                                      <div className="truncate text-xs font-semibold">{match.title || "Matched product"}</div>
                                      <div className="text-[11px] text-muted-foreground">
                                        {match.source || "Unknown source"}
                                        {match.price ? ` • ${match.price}` : ""}
                                      </div>
                                    </div>
                                  </a>
                                ) : (
                                  <div className="mt-2 rounded border border-dashed border-border px-2 py-1 text-[11px] text-muted-foreground">
                                    Product lookup pending / no strong match yet.
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </section>
                  </div>
                  );
                })}
              </div>
            )}
          </section>
        </section>
      )}
    </div>
  );
}
