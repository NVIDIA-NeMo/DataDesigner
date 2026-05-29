/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";

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
  image?: ReactNode;
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

/**
 * Product CSS for the blog index, injected by <BlogGrid> rather than loaded via
 * docs.yml `css:`. `css` is a theme-owned field, so under `global-theme: nvidia`
 * Fern drops any local `css:` list at publish — styling has to travel with the
 * component instead. See the note in fern/docs.yml.
 */
const BLOG_CARD_CSS = `
.blog-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.25rem;
  margin: 1.5rem 0 2rem;
}
.blog-card {
  display: flex;
  flex-direction: column;
  border-radius: 12px;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  background: var(--grayscale-1, transparent);
  overflow: hidden;
  text-decoration: none !important;
  color: inherit !important;
  transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
}
.blog-card:hover {
  border-color: var(--accent, #76b900); /* theme accent, falls back to NVIDIA green */
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(118, 185, 0, 0.08);
}
.blog-card__media {
  aspect-ratio: 16 / 9;
  width: 100%;
  background: var(--grayscale-3, rgba(128, 128, 128, 0.06));
  overflow: hidden;
}
/* Reset prose margins on the wrappers Fern injects around <img> (e.g. the rmiz
   click-to-zoom <span> shells), which otherwise push the image off the card top. */
.blog-card__media > *,
.blog-card__media > * > * {
  display: block;
  width: 100%;
  height: 100%;
  margin: 0 !important;
  padding: 0 !important;
}
.blog-card__media img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
  display: block;
  margin: 0 !important;
}
/* Defeat Fern's auto-injected click-to-zoom (rmiz): the whole card is already a
   link, so clicks should navigate, not open a lightbox. */
.blog-card__media [data-rmiz],
.blog-card__media [data-rmiz-content],
.blog-card__media img {
  pointer-events: none;
}
.blog-card__placeholder {
  width: 100%;
  height: 100%;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.blog-card__placeholder::before {
  content: "";
  position: absolute;
  inset: -10%;
  background:
    radial-gradient(ellipse 60% 50% at 30% 30%, rgba(255, 255, 255, 0.18), transparent 60%),
    radial-gradient(ellipse 40% 35% at 75% 80%, rgba(0, 0, 0, 0.22), transparent 55%);
  pointer-events: none;
}
.blog-card__monogram {
  position: relative;
  font-size: clamp(3.5rem, 9vw, 5rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  color: rgba(255, 255, 255, 0.92);
  text-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
  font-family: -apple-system, system-ui, sans-serif;
  line-height: 1;
}
.blog-card__body {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem 1.125rem 1.125rem;
  flex: 1;
}
.blog-card__date {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--grayscale-9, #6b7280);
}
.blog-card__title {
  margin: 0;
  font-size: 1.0625rem;
  font-weight: 600;
  line-height: 1.3;
  color: var(--grayscale-12, inherit);
}
.blog-card__description {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
  color: var(--grayscale-11, #4b5563);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.blog-card__byline {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: auto;
  padding-top: 0.75rem;
  border-top: 1px solid var(--grayscale-a4, rgba(128, 128, 128, 0.1));
}
.blog-card__avatars {
  display: flex;
  align-items: center;
}
.blog-card__avatar {
  border-radius: 999px;
  border: 2px solid var(--grayscale-1, #fff);
  margin-left: -6px;
  background: var(--grayscale-3, #ddd);
}
.blog-card__avatar:first-child {
  margin-left: 0;
}
.blog-card__authors {
  font-size: 0.8125rem;
  color: var(--grayscale-10, #6b7280);
  font-weight: 500;
}
`;

export function BlogGrid({ children }: { children: ReactNode }) {
  return (
    <div className="blog-grid">
      {/* static CSS string literal (no user input) — safe to inject as raw HTML */}
      <style dangerouslySetInnerHTML={{ __html: BLOG_CARD_CSS }} />
      {children}
    </div>
  );
}
