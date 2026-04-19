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

type StepBox = {
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  left?: number;
  top?: number;
  width?: number;
  height?: number;
};

type ClothingStepItem = {
  category?: string;
  confidence?: number;
  visibility?: string;
  description?: string;
  bounding_box?: StepBox;
};

type YoloDebugInfo = {
  status?: string;
  prediction_status?: string;
  error?: string;
  model_version?: string;
  detection_count?: number;
  poll_count?: number;
  poll_error_http_status?: number;
  start_attempts?: Array<{
    wait_seconds?: number;
    http_status?: number;
  }>;
};

type ClothingExtractionStep = {
  photo_id: string;
  reason?: string;
  postprocess_source?: string;
  yolo_requested_source?: string;
  yolo_requested_classes?: string[];
  gpt_identified_classes?: string[];
  yolo_world_detections?: Array<{
    label?: string;
    confidence?: number;
    bbox?: number[];
  }>;
  gpt_cleaned_items?: ClothingStepItem[];
  visible_items?: ClothingStepItem[];
  inserted_item_count?: number;
  yolo_debug?: YoloDebugInfo;
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
    per_photo_steps?: ClothingExtractionStep[];
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
  closet_item_key?: string;
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

type BackfillFavoritesResponse = {
  phia_id: string;
  collection_id: string;
  source: string;
  requested_count: number;
  attempted_count: number;
  added_count: number;
  failed_count: number;
  results: Array<{
    product_url: string;
    ok: boolean;
    message: string;
  }>;
};

const POLL_MS = 1200;
type PhotoBreakdownViewMode = "original" | "vlm" | "final";

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
  bb: StepBox | null | undefined
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

function getYoloBoxStyle(
  bbox: number[] | null | undefined,
  imageWidth: number | null | undefined,
  imageHeight: number | null | undefined
): React.CSSProperties {
  if (!bbox || bbox.length < 4) {
    return { display: "none" };
  }

  const rawX1 = Number(bbox[0] ?? 0);
  const rawY1 = Number(bbox[1] ?? 0);
  const rawX2 = Number(bbox[2] ?? 0);
  const rawY2 = Number(bbox[3] ?? 0);
  const safeW = Number(imageWidth ?? 0);
  const safeH = Number(imageHeight ?? 0);

  let x1 = rawX1;
  let y1 = rawY1;
  let x2 = rawX2;
  let y2 = rawY2;

  if (safeW > 0 && safeH > 0) {
    x1 /= safeW;
    x2 /= safeW;
    y1 /= safeH;
    y2 /= safeH;
  }

  x1 = Math.max(0, Math.min(1, x1));
  y1 = Math.max(0, Math.min(1, y1));
  x2 = Math.max(0, Math.min(1, x2));
  y2 = Math.max(0, Math.min(1, y2));

  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);

  return {
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${width * 100}%`,
    height: `${height * 100}%`,
  };
}

export function PipelineDashboard({ backendUrl }: Props) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busySelectClusterId, setBusySelectClusterId] = useState<string | null>(null);
  const [visibleSerpUrls, setVisibleSerpUrls] = useState<string[]>([]);
  const [carouselState, setCarouselState] = useState<{ key: string; index: number }>({
    key: "",
    index: 0,
  });
  const [photoViewState, setPhotoViewState] = useState<{ key: string; mode: PhotoBreakdownViewMode }>({
    key: "",
    mode: "final",
  });
  const [isBackfillBusy, setIsBackfillBusy] = useState(false);
  const [backfillNotice, setBackfillNotice] = useState<string | null>(null);
  const [backfillError, setBackfillError] = useState<string | null>(null);

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
  const closetProducts = useMemo(() => {
    const deduped = new Map<string, { item: ClothingItem; match: ClothingItem["best_match"] }>();
    for (const item of chosenFacePhotoRows.flatMap(({ items }) => items)) {
      const match = item.best_match;
      if (!match?.link) continue;
      const canonicalKey = (item.closet_item_key ?? "").trim() || item.id;
      const existing = deduped.get(canonicalKey);
      if (!existing) {
        deduped.set(canonicalKey, { item, match });
        continue;
      }
      const currentScore = item.best_match_confidence ?? 0;
      const previousScore = existing.item.best_match_confidence ?? 0;
      if (currentScore > previousScore) {
        deduped.set(canonicalKey, { item, match });
      }
    }
    return Array.from(deduped.values());
  }, [chosenFacePhotoRows]);
  const closetProductUrls = useMemo(() => {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const { match } of closetProducts) {
      const url = (match?.link ?? "").trim();
      if (!url || seen.has(url)) continue;
      seen.add(url);
      out.push(url);
    }
    return out;
  }, [closetProducts]);

  const clothingStepByPhoto = useMemo(() => {
    const map = new Map<string, ClothingExtractionStep>();
    const steps = detail?.debug?.clothing_extraction?.per_photo_steps ?? [];
    for (const step of steps) {
      if (step?.photo_id) {
        map.set(step.photo_id, step);
      }
    }
    return map;
  }, [detail?.debug?.clothing_extraction?.per_photo_steps]);

  const carouselKey = `${detail?.id ?? ""}:${selectedCluster?.id ?? ""}`;
  const carouselMaxIndex = Math.max(0, chosenFacePhotoRows.length - 1);
  const chosenFacePhotoIndex =
    carouselState.key === carouselKey ? Math.min(carouselState.index, carouselMaxIndex) : 0;
  const activeChosenFaceRow = chosenFacePhotoRows[chosenFacePhotoIndex] ?? null;
  const activePhotoViewMode: PhotoBreakdownViewMode = photoViewState.key === carouselKey ? photoViewState.mode : "final";

  const setChosenFacePhotoIndex = (nextIndex: number) => {
    const clamped = Math.max(0, Math.min(carouselMaxIndex, nextIndex));
    setCarouselState({ key: carouselKey, index: clamped });
  };
  const setActivePhotoViewMode = (mode: PhotoBreakdownViewMode) => {
    setPhotoViewState({ key: carouselKey, mode });
  };

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

  const handleBackfillFavorites = useCallback(async () => {
    if (closetProductUrls.length === 0) {
      setBackfillError("No product links found in closet yet.");
      setBackfillNotice(null);
      return;
    }
    setIsBackfillBusy(true);
    setBackfillError(null);
    setBackfillNotice(null);
    try {
      const res = await fetch(`${backendUrl}/api/phia/backfill-favorites`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_urls: closetProductUrls,
          collection_id: "all_favorites",
        }),
      });
      const payload = (await res.json()) as BackfillFavoritesResponse | { detail?: string };
      if (!res.ok) {
        throw new Error((payload as { detail?: string }).detail || `Backfill HTTP ${res.status}`);
      }
      const data = payload as BackfillFavoritesResponse;
      setBackfillNotice(`Added ${data.added_count}/${data.attempted_count} products to ${data.collection_id}.`);
      if (data.failed_count > 0) {
        const firstFailure = data.results.find((entry) => !entry.ok)?.message;
        setBackfillError(
          firstFailure
            ? `Some products failed: ${firstFailure}`
            : `Some products failed (${data.failed_count}/${data.attempted_count}).`
        );
      }
    } catch (e) {
      setBackfillError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsBackfillBusy(false);
    }
  }, [backendUrl, closetProductUrls]);

  return (
    <div className="phia-dashboard space-y-6">
      <section className="phia-panel p-5">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Jobs</h2>
        <div className="phia-scroll flex gap-3 overflow-x-auto pb-1">
          {jobs.map((job) => (
            <button
              key={job.id}
              type="button"
              onClick={() => {
                setSelectedJobId(job.id);
              }}
              className={`phia-soft-button min-w-[220px] rounded-2xl border px-3 py-2 text-left transition-colors ${
                selectedJobId === job.id
                  ? "border-accent bg-accent/14 shadow-[0_14px_24px_-20px_rgba(47,99,216,0.9)]"
                  : "border-border bg-white/75 hover:bg-white"
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
        <section className="phia-panel p-5 text-sm text-muted-foreground">
          Waiting for selected job...
        </section>
      ) : (
        <section className="phia-panel space-y-6 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-xl font-semibold tracking-[-0.02em]">Selected Job {detail.id.slice(0, 8)}</h2>
            <span className="phia-chip px-3 py-1 text-xs font-semibold uppercase tracking-[0.13em]">
              {detail.status}
            </span>
            {detail.error ? <span className="text-sm text-destructive">{detail.error}</span> : null}
            {error ? <span className="text-sm text-destructive">{error}</span> : null}
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <section className="phia-subpanel space-y-3 p-4 xl:col-span-1">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Uploaded Photos</h3>
              <div className="phia-scroll h-[520px] space-y-3 overflow-y-auto pr-1">
                {detail.photos.map((photo) => {
                  const detections = detectionsByPhoto.get(photo.id) ?? [];
                  const aspect =
                    (photo.width ?? 0) > 0 && (photo.height ?? 0) > 0
                      ? (photo.width ?? 1) / (photo.height ?? 1)
                      : 1;
                  return (
                    <div key={photo.id} className="rounded-2xl border border-border bg-white/80 p-2.5">
                      <div className="grid grid-cols-[96px_minmax(0,1fr)] gap-3">
                        <img src={photo.url} alt="uploaded" className="h-20 w-full rounded-xl object-cover shadow-sm" />
                        <div>
                          <div className="mb-1 flex items-center justify-between">
                            <span className="font-mono text-[11px] text-muted-foreground">{photo.id.slice(0, 6)}</span>
                            <span className="text-xs font-semibold">{detections.length} faces</span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {detections.map((det) => (
                              <div
                                key={det.id}
                                className="relative h-10 w-10 overflow-hidden rounded-full border border-border bg-card shadow-sm"
                              >
                                <img src={photo.url} alt="face crop" style={getFaceCropStyle(det.bbox, aspect, 40)} />
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

            <section className="phia-subpanel space-y-3 p-4 xl:col-span-1">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Aggregated Faces</h3>
              <div className="phia-scroll h-[520px] overflow-y-auto pr-1">
                <div className="grid grid-cols-3 gap-3 lg:grid-cols-4">
                {detail.clusters.map((cluster) => (
                  <button
                    key={cluster.id}
                    type="button"
                    disabled={busySelectClusterId !== null}
                    onClick={() => void chooseCluster(cluster.id)}
                    className={`phia-soft-button rounded-2xl border bg-white/80 p-2 text-center transition-colors ${
                      detail.selected_cluster_id === cluster.id ? "border-accent bg-accent/12" : "border-border"
                    }`}
                  >
                    <div className="relative mx-auto h-12 w-12 overflow-hidden rounded-full border border-border bg-card shadow-sm">
                      <img
                        src={cluster.source_url}
                        alt="aggregated face"
                        style={getFaceCropStyle(cluster.rep_bbox, cluster.rep_aspect_ratio ?? 1, 48)}
                      />
                    </div>
                    <div className="mt-1 text-xs font-semibold">{cluster.member_count}</div>
                    <div className="text-[10px] text-muted-foreground">photos</div>
                  </button>
                ))}
                </div>
              </div>
            </section>

            <section className="phia-subpanel space-y-3 p-4 xl:col-span-1">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                {`Google Image Response For "${personName}"`}
              </h3>
              <div className="phia-scroll h-[520px] overflow-y-auto pr-1">
                <div className="grid grid-cols-3 gap-2 lg:grid-cols-4">
                  {visibleSerpUrls.slice(0, 24).map((url, idx) => (
                    <a
                      key={`${idx}-${url}`}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`overflow-hidden rounded-xl border bg-white/80 shadow-sm ${
                        selectedMatchedUrlKeySet.has(imageKey(url))
                          ? "border-4 border-emerald-600"
                          : "border-border"
                      }`}
                    >
                      <img
                        src={url}
                        alt=""
                        className="h-12 w-full object-cover"
                        onError={() => setVisibleSerpUrls((prev) => prev.filter((u) => u !== url))}
                      />
                    </a>
                  ))}
                </div>
              </div>
            </section>
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
            <section className="phia-subpanel p-4">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
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

            <section className="phia-subpanel p-4 text-center">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Chosen Face</h3>
              {selectedCluster ? (
                <>
                  <div className="relative mx-auto mt-2 h-20 w-20 overflow-hidden rounded-full border border-border bg-card shadow-sm">
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
                <div className="mt-3 rounded-xl border border-dashed border-border bg-white/75 px-3 py-4 text-sm text-muted-foreground">
                  {shouldAskUserConfirm ? "Waiting on user response" : "No face chosen yet"}
                </div>
              )}
            </section>
          </div>

          <section className="phia-subpanel space-y-3 p-4">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Chosen Face Photo Breakdown
            </h3>
            {!selectedCluster ? (
              <div className="rounded-xl border border-dashed border-border bg-white/75 px-3 py-4 text-sm text-muted-foreground">
                Choose a face cluster to see photo/item/product breakdown.
              </div>
            ) : chosenFacePhotoRows.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border bg-white/75 px-3 py-4 text-sm text-muted-foreground">
                No photos found for the chosen face yet.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between rounded-2xl border border-border bg-white/75 px-3 py-2">
                  <button
                    type="button"
                    onClick={() => setChosenFacePhotoIndex(chosenFacePhotoIndex - 1)}
                    disabled={chosenFacePhotoIndex === 0}
                    className="phia-soft-button rounded-xl border border-border bg-background px-2 py-1 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    ← Previous
                  </button>
                  <div className="text-xs font-semibold text-muted-foreground">
                    Photo {chosenFacePhotoIndex + 1} of {chosenFacePhotoRows.length}
                  </div>
                  <button
                    type="button"
                    onClick={() => setChosenFacePhotoIndex(chosenFacePhotoIndex + 1)}
                    disabled={chosenFacePhotoIndex >= chosenFacePhotoRows.length - 1}
                    className="phia-soft-button rounded-xl border border-border bg-background px-2 py-1 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Next →
                  </button>
                </div>

                {activeChosenFaceRow ? (
                  (() => {
                    const { photo, items } = activeChosenFaceRow;
                    const photoAspect =
                      photo.width && photo.height && photo.height > 0
                        ? photo.width / photo.height
                        : 1;
                    const step = clothingStepByPhoto.get(photo.id);
                    const yoloOutput = step?.yolo_world_detections ?? [];
                    const gptCleaned = step?.gpt_cleaned_items ?? [];
                    const finalBoxes: Array<{ description?: string; bounding_box?: StepBox }> = gptCleaned.map((item) => ({
                      description: item.description,
                      bounding_box: item.bounding_box,
                    }));

                    const modeLabel =
                      activePhotoViewMode === "original"
                        ? "Original Image"
                        : activePhotoViewMode === "vlm"
                          ? "VLM output"
                          : "Final Image";
                    const modeCount =
                      activePhotoViewMode === "original"
                        ? 0
                        : activePhotoViewMode === "vlm"
                          ? yoloOutput.length
                          : finalBoxes.length;
                    const legendEntries =
                      activePhotoViewMode === "vlm"
                        ? yoloOutput.map((det, idx) => ({
                            index: idx + 1,
                            text: `${(det.label || "unknown").toLowerCase()} ${(Number(det.confidence || 0) * 100).toFixed(0)}%`,
                          }))
                        : activePhotoViewMode === "final"
                          ? finalBoxes.map((item, idx) => ({
                              index: idx + 1,
                              text: item.description || "item",
                            }))
                          : [];

                    return (
                      <div
                        key={photo.id}
                        className="phia-scroll grid h-[780px] gap-3 overflow-y-auto rounded-2xl border border-border bg-white/80 p-3 lg:grid-cols-[180px_minmax(0,1fr)]"
                      >
                        <aside className="phia-subpanel space-y-2 p-2.5">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Image View</div>
                          <button
                            type="button"
                            onClick={() => setActivePhotoViewMode("original")}
                            className={`phia-soft-button w-full rounded-xl border px-2 py-2 text-left text-xs font-semibold ${
                              activePhotoViewMode === "original" ? "border-accent bg-accent/12" : "border-border bg-card"
                            }`}
                          >
                            Original Image
                          </button>
                          <button
                            type="button"
                            onClick={() => setActivePhotoViewMode("vlm")}
                            className={`phia-soft-button w-full rounded-xl border px-2 py-2 text-left text-xs font-semibold ${
                              activePhotoViewMode === "vlm" ? "border-accent bg-accent/12" : "border-border bg-card"
                            }`}
                          >
                            VLM output
                          </button>
                          <button
                            type="button"
                            onClick={() => setActivePhotoViewMode("final")}
                            className={`phia-soft-button w-full rounded-xl border px-2 py-2 text-left text-xs font-semibold ${
                              activePhotoViewMode === "final" ? "border-accent bg-accent/12" : "border-border bg-card"
                            }`}
                          >
                            Final Image
                          </button>
                          <div className="pt-1 text-[11px] text-muted-foreground">
                            <div>photo {photo.id.slice(0, 8)}</div>
                            <div className="mt-1">{modeLabel}</div>
                            <div>{modeCount} boxes</div>
                            <div className="mt-1">YOLO: {yoloOutput.length}</div>
                          </div>
                        </aside>

                        <section className="space-y-3">
                          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px] md:items-start">
                            <div
                              className="relative mx-auto w-full max-w-[360px] overflow-hidden rounded-2xl border border-border bg-background"
                              style={{ aspectRatio: photoAspect }}
                            >
                              <img src={photo.url} alt="photo breakdown image" className="absolute inset-0 h-full w-full object-contain" />
                              {activePhotoViewMode === "vlm"
                                ? yoloOutput.map((det, idx) => (
                                    <div
                                      key={`${photo.id}-vlm-box-${idx}`}
                                      className="pointer-events-none absolute"
                                      style={getYoloBoxStyle(det.bbox, photo.width, photo.height)}
                                    >
                                      <div className="h-full w-full border-2 border-orange-500" />
                                      <span className="absolute left-0 top-0 rounded-br bg-orange-500 px-1.5 py-0.5 text-[10px] font-semibold text-black">
                                        {idx + 1}
                                      </span>
                                    </div>
                                  ))
                                : null}
                              {activePhotoViewMode === "final"
                                ? finalBoxes.map((item, idx) => (
                                    <div
                                      key={`${photo.id}-final-box-${idx}`}
                                      className="pointer-events-none absolute"
                                      style={getItemBoxStyle(item.bounding_box)}
                                    >
                                      <div className="h-full w-full border-2 border-red-500" />
                                      <span className="absolute left-0 top-0 rounded-br bg-red-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                                        {idx + 1}
                                      </span>
                                    </div>
                                  ))
                                : null}
                            </div>

                            {activePhotoViewMode !== "original" ? (
                              <aside className="phia-subpanel p-2.5">
                                <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                                  {activePhotoViewMode === "vlm" ? "VLM Labels" : "Final Labels"}
                                </div>
                                <div className="phia-scroll mt-2 max-h-[360px] space-y-1 overflow-y-auto pr-1">
                                  {legendEntries.length === 0 ? (
                                    <div className="text-xs text-muted-foreground">No detections yet.</div>
                                  ) : (
                                    legendEntries.map((entry) => (
                                      <div key={`${photo.id}-legend-${entry.index}`} className="flex items-start gap-2 rounded-xl border border-border bg-card px-2 py-1">
                                        <span
                                          className={`mt-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded text-[10px] font-semibold ${
                                            activePhotoViewMode === "vlm"
                                              ? "bg-orange-500 text-black"
                                              : "bg-red-600 text-white"
                                          }`}
                                        >
                                          {entry.index}
                                        </span>
                                        <span className="text-xs text-foreground">{entry.text}</span>
                                      </div>
                                    ))
                                  )}
                                </div>
                              </aside>
                            ) : null}
                          </div>

                          <section>
                            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                              Products In This Photo
                            </div>
                            <div className="phia-scroll h-64 overflow-y-auto pr-1">
                              {items.length === 0 ? (
                                <div className="rounded-xl border border-dashed border-border bg-background px-3 py-4 text-sm text-muted-foreground">
                                  No extracted items yet.
                                </div>
                              ) : (
                                (() => {
                                  const matchedProducts = items
                                    .map((item) => ({ item, match: item.best_match }))
                                    .filter((entry) => !!entry.match?.link);

                                  if (matchedProducts.length === 0) {
                                    return (
                                      <div className="rounded-xl border border-dashed border-border px-2 py-1 text-[11px] text-muted-foreground">
                                        Serp lookup pending / no strong match yet.
                                      </div>
                                    );
                                  }

                                  return (
                                    <div className="phia-scroll flex gap-3 overflow-x-auto pb-2">
                                      {matchedProducts.map(({ item, match }) => (
                                        <a
                                          key={`${item.id}-${match?.link || "match"}`}
                                          href={match?.link}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="phia-soft-button w-36 shrink-0 rounded-xl border border-border bg-background p-2"
                                        >
                                          {match?.thumbnail ? (
                                            <img src={match.thumbnail} alt={match.title || "matched product"} className="h-32 w-32 rounded object-cover" />
                                          ) : (
                                            <div className="h-32 w-32 rounded bg-muted" />
                                          )}
                                          <div className="mt-2 truncate text-xs font-semibold">{match?.title || "Matched product"}</div>
                                          <div className="text-[11px] text-muted-foreground">
                                            {match?.price || "Price unavailable"}
                                          </div>
                                        </a>
                                      ))}
                                    </div>
                                  );
                                })()
                              )}
                            </div>
                          </section>
                        </section>
                      </div>
                    );
                  })()
                ) : null}

                <section className="phia-subpanel p-4">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <div className="text-sm font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                      Closet
                    </div>
                    <button
                      type="button"
                      onClick={() => void handleBackfillFavorites()}
                      disabled={isBackfillBusy || closetProductUrls.length === 0}
                      className="phia-soft-button rounded-xl border border-border bg-background px-3 py-1.5 text-xs font-semibold text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {isBackfillBusy ? "Backfilling..." : "backfill favorites collection"}
                    </button>
                  </div>
                  {backfillNotice ? <div className="mb-2 text-xs text-emerald-700">{backfillNotice}</div> : null}
                  {backfillError ? <div className="mb-2 text-xs text-destructive">{backfillError}</div> : null}
                  {closetProducts.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-border bg-background px-3 py-4 text-sm text-muted-foreground">
                      No product items found yet.
                    </div>
                  ) : (
                    <div className="phia-scroll max-h-[720px] overflow-y-auto pr-1">
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                        {closetProducts.map(({ item, match }) => (
                          <a
                            key={`closet-${item.id}-${match?.link || "match"}`}
                            href={match?.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="phia-soft-button rounded-xl border border-border bg-background p-2"
                          >
                            {match?.thumbnail ? (
                              <img src={match.thumbnail} alt={match.title || item.description} className="h-28 w-full rounded object-cover" />
                            ) : (
                              <div className="h-28 w-full rounded bg-muted" />
                            )}
                            <div className="mt-2 truncate text-xs font-semibold">{match?.title || item.description}</div>
                            <div className="text-[11px] text-muted-foreground">
                              {match?.price || "Price unavailable"}
                            </div>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </section>
              </div>
            )}
          </section>
        </section>
      )}
    </div>
  );
}
