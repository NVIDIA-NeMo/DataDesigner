/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { authors as REGISTRY } from "./devnotes/authors-data";

/**
 * Site basepath. Mirrors `instances[0].custom-domain` path in fern/docs.yml.
 * Custom MDX components bypass Fern's link rewriter, so the card's `href`
 * needs the prefix manually to avoid 404s under basepath-aware routing.
 *
 * Image paths are NOT prefixed — they should be passed in as ES-module imports
 * from MDX (e.g. `import hero from "@/assets/foo/hero.png"`), which the
 * bundler resolves to the correct URL in both dev and production.
 */
const BASEPATH = "/nemo/datadesigner";

/** Prepend BASEPATH to a root-relative path if not already present. */
function withBasepath(path: string): string {
  if (!path.startsWith("/")) return path;
  if (path.startsWith(BASEPATH + "/") || path === BASEPATH) return path;
  return BASEPATH + path;
}

/**
 * BlogCard — index card for a dev note / blog post.
 *
 * Renders a clickable tile with: optional hero image, ALL-CAPS date eyebrow,
 * title, description, and an author byline (avatar stack + first author + "+N").
 *
 * Designed for the dev-notes landing index — Fern's built-in <Card> only does
 * icon + title + description, which made every card visually identical.
 *
 * Usage in MDX (inside <BlogGrid>):
 *
 *   import { BlogCard, BlogGrid } from "@/components/BlogCard";
 *
 *   <BlogGrid>
 *     <BlogCard
 *       href="/dev-notes/push-datasets-to-hugging-face-hub"
 *       title="Push Datasets to Hugging Face Hub"
 *       description="Call .push_to_hub() and ship a generated dataset…"
 *       date="Apr 16, 2026"
 *       authors={["nmulepati", "davanstrien"]}
 *       image="/assets/push-datasets-to-hugging-face-hub/push-to-hub-hero.png"
 *     />
 *   </BlogGrid>
 */

export interface BlogCardProps {
  href: string;
  title: string;
  description: string;
  date: string;
  authors?: string[];
  /**
   * Optional hero image element. Pass an `<img>` JSX node from MDX so Fern's
   * MDX rewriter resolves the src to the correct dev/prod path (raw string
   * paths bypass the rewriter and 404 in dev). Falls back to a deterministic
   * hash-based gradient + monogram when omitted.
   *
   *   <BlogCard image={<img src="/assets/foo/hero.png" alt="" />} … />
   */
  image?: React.ReactNode;
}

/** Deterministic hash → number ∈ [0, 360). Same input → same color. */
function hashHue(input: string): number {
  let h = 5381;
  for (let i = 0; i < input.length; i++) {
    h = ((h << 5) + h + input.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % 360;
}

/** Build a 2-stop diagonal gradient that reads well in both light/dark.
 * Hue is constrained to a band that pairs with NVIDIA green (avoid muddy
 * yellows by skipping 40-90°). */
function placeholderGradient(seed: string): string {
  let hue = hashHue(seed);
  if (hue >= 40 && hue < 90) hue = (hue + 60) % 360;
  const a = `hsl(${hue} 55% 38%)`;
  const b = `hsl(${(hue + 35) % 360} 60% 22%)`;
  return `linear-gradient(135deg, ${a} 0%, ${b} 100%)`;
}

/** First grapheme of the title (works for "🎨 Title" too). */
function monogramOf(title: string): string {
  // Strip leading non-letter punctuation/whitespace then take 1 char.
  const trimmed = title.replace(/^[^\p{L}\p{N}]+/u, "");
  return Array.from(trimmed)[0]?.toUpperCase() ?? "·";
}

export function BlogCard({
  href,
  title,
  description,
  date,
  authors = [],
  image,
}: BlogCardProps) {
  const validAuthors = authors.map((id) => REGISTRY[id]).filter(Boolean);
  const primary = validAuthors[0];
  const extra = validAuthors.length - 1;

  return (
    <a className="blog-card" href={withBasepath(href)}>
      <div className="blog-card__media">
        {image ? (
          image
        ) : (
          <div
            className="blog-card__placeholder"
            style={{ background: placeholderGradient(href) }}
            aria-hidden="true"
          >
            <span className="blog-card__monogram">{monogramOf(title)}</span>
          </div>
        )}
      </div>
      <div className="blog-card__body">
        <span className="blog-card__date">{date}</span>
        <h3 className="blog-card__title">{title}</h3>
        <p className="blog-card__description">{description}</p>
        {primary && (
          <div className="blog-card__byline">
            <div className="blog-card__avatars">
              {validAuthors.slice(0, 3).map((a, i) => (
                <img
                  key={i}
                  className="blog-card__avatar"
                  src={a.avatar}
                  alt=""
                  width={20}
                  height={20}
                />
              ))}
            </div>
            <span className="blog-card__authors">
              {primary.name}
              {extra > 0 ? ` +${extra}` : ""}
            </span>
          </div>
        )}
      </div>
    </a>
  );
}

export function BlogGrid({ children }: { children: React.ReactNode }) {
  return <div className="blog-grid">{children}</div>;
}
