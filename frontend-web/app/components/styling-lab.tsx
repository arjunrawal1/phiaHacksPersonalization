"use client";
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useState } from "react";

type Props = {
  backendUrl: string;
};

type JobSummary = {
  id: string;
};

type AutoRunVariant = {
  variant_index: number;
  output_url: string;
  is_best?: boolean;
};

type AutoRun = {
  photo_id: string;
  status: "waiting" | "queued" | "running" | "completed" | "skipped" | "failed";
  best_variant_index?: number | null;
  variants: AutoRunVariant[];
};

type AutoRunsResponse = {
  job_id: string;
  runs: AutoRun[];
};

type JobDetail = {
  items?: Array<{
    photo_id: string;
    best_match?: {
      title?: string | null;
      price?: string | null;
      link?: string | null;
      thumbnail?: string | null;
    } | null;
  }>;
};

type MatchedProduct = {
  title: string;
  price: string;
  link: string;
  thumbnail: string;
};

type ResultPair = {
  products: MatchedProduct[];
  photoshootUrl: string;
};

const POLL_MS = 1500;

export function StylingLab({ backendUrl }: Props) {
  const [pairs, setPairs] = useState<ResultPair[]>([]);
  const [rowIndex, setRowIndex] = useState(0);

  const load = useCallback(async () => {
    try {
      const jobsRes = await fetch(`${backendUrl}/api/jobs`, { cache: "no-store" });
      if (!jobsRes.ok) return;
      const jobs = (await jobsRes.json()) as JobSummary[];
      const latestJobId = jobs[0]?.id;
      if (!latestJobId) {
        setPairs([]);
        return;
      }

      const [runsRes, detailRes] = await Promise.all([
        fetch(`${backendUrl}/api/styling/auto-runs?job_id=${encodeURIComponent(latestJobId)}`, {
          cache: "no-store",
        }),
        fetch(`${backendUrl}/api/jobs/${encodeURIComponent(latestJobId)}`, {
          cache: "no-store",
        }),
      ]);
      if (!runsRes.ok || !detailRes.ok) return;

      const payload = (await runsRes.json()) as AutoRunsResponse;
      const detail = (await detailRes.json()) as JobDetail;

      const itemsByPhotoId = new Map<string, Array<NonNullable<JobDetail["items"]>[number]>>();
      for (const item of detail.items ?? []) {
        const current = itemsByPhotoId.get(item.photo_id) ?? [];
        current.push(item);
        itemsByPhotoId.set(item.photo_id, current);
      }

      const nextPairs = payload.runs
        .filter((run) => run.status === "completed")
        .map((run) => {
          let photoshootUrl = "";
          const bestByFlag = run.variants.find((variant) => variant.is_best);
          if (bestByFlag?.output_url) {
            photoshootUrl = bestByFlag.output_url;
          } else if (typeof run.best_variant_index === "number") {
            const bestByIndex = run.variants.find(
              (variant) => variant.variant_index === run.best_variant_index
            );
            photoshootUrl = bestByIndex?.output_url ?? "";
          } else {
            photoshootUrl = run.variants[0]?.output_url ?? "";
          }

          const itemsForPhoto = itemsByPhotoId.get(run.photo_id) ?? [];
          const products: MatchedProduct[] = [];
          const seen = new Set<string>();
          for (const item of itemsForPhoto) {
            const link = (item.best_match?.link || "").trim();
            if (!link || seen.has(link)) continue;
            seen.add(link);
            products.push({
              title: (item.best_match?.title || "Matched product").trim(),
              price: (item.best_match?.price || "Price unavailable").trim(),
              link,
              thumbnail: (item.best_match?.thumbnail || "").trim(),
            });
            if (products.length >= 8) {
              break;
            }
          }

          return {
            photoshootUrl,
            products,
          };
        })
        .filter((pair) => !!pair.photoshootUrl);

      setPairs(nextPairs);
    } catch {
      // keep current grid state on transient polling failures
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

  useEffect(() => {
    setRowIndex((prev) => {
      if (pairs.length === 0) return 0;
      if (prev < 0) return 0;
      if (prev >= pairs.length) return pairs.length - 1;
      return prev;
    });
  }, [pairs.length]);

  const activePair = pairs[rowIndex] ?? null;

  return (
    <section className="phia-panel p-4 sm:p-6">
      {pairs.length > 0 ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between rounded-2xl border border-border bg-white/75 px-3 py-2">
            <button
              type="button"
              onClick={() => setRowIndex((prev) => Math.max(0, prev - 1))}
              disabled={rowIndex === 0}
              className="phia-soft-button rounded-xl border border-border bg-background px-2 py-1 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            >
              ← Previous
            </button>
            <div className="text-xs font-semibold text-muted-foreground">
              Look {rowIndex + 1} of {pairs.length}
            </div>
            <button
              type="button"
              onClick={() => setRowIndex((prev) => Math.min(pairs.length - 1, prev + 1))}
              disabled={rowIndex >= pairs.length - 1}
              className="phia-soft-button rounded-xl border border-border bg-background px-2 py-1 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50"
            >
              Next →
            </button>
          </div>

          {activePair ? (
            <div className="overflow-hidden rounded-xl border border-border bg-card p-2">
              <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_220px]">
                <div className="rounded-lg bg-muted/35 p-1">
                  <img
                    src={activePair.photoshootUrl}
                    alt={`Photoshoot ${rowIndex + 1}`}
                    className="h-[420px] w-full rounded-md object-contain"
                  />
                </div>
                <div className="rounded-lg bg-muted/35 p-1">
                  {activePair.products.length > 0 ? (
                    <div className="phia-scroll max-h-[420px] space-y-2 overflow-y-auto pr-1">
                      {activePair.products.map((product, productIdx) => (
                        <a
                          key={`${product.link}-${productIdx}`}
                          href={product.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="phia-soft-button block rounded-md border border-border bg-background p-2"
                        >
                          {product.thumbnail ? (
                            <img
                              src={product.thumbnail}
                              alt={product.title || `Product ${rowIndex + 1}-${productIdx + 1}`}
                              className="h-[150px] w-full rounded object-cover"
                            />
                          ) : (
                            <div className="h-[150px] w-full rounded bg-muted" />
                          )}
                          <div className="mt-2 min-w-0">
                            <div className="truncate text-xs font-semibold text-foreground">
                              {product.title || "Matched product"}
                            </div>
                            <div className="text-[11px] text-muted-foreground">
                              {product.price || "Price unavailable"}
                            </div>
                          </div>
                        </a>
                      ))}
                    </div>
                  ) : (
                    <div className="flex h-[420px] items-center justify-center rounded-md border border-dashed border-border text-xs text-muted-foreground">
                      Product image unavailable
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
