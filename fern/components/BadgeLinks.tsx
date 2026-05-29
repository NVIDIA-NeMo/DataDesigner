/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Horizontal badge row (shields.io / img.shields.io style images wrapped in
 * anchors). Uses a flex container so badges sit side-by-side instead of
 * stacking with Fern's default external-link icon overlay.
 *
 * Pass `badges` explicitly — required so we never accidentally ship
 * placeholder URLs to production.
 *
 * Usage in MDX:
 *   import { BadgeLinks } from "@/components/BadgeLinks";
 *
 *   <BadgeLinks
 *     badges={[
 *       { href: "https://github.com/NVIDIA-NeMo/DataDesigner",
 *         src:  "https://img.shields.io/badge/github-repo-952fc6?logo=github",
 *         alt:  "GitHub" },
 *     ]}
 *   />
 */
export type BadgeItem = {
  href: string;
  src: string;
  alt: string;
};

/**
 * Hide Fern's redundant external-link icon on badge anchors (the shields already
 * read as links). Injected here rather than via docs.yml `css:` — `css` is
 * theme-owned, so under `global-theme: nvidia` a local `css:` list is dropped at
 * publish. See fern/docs.yml.
 */
const BADGE_LINKS_CSS = `
.badge-links .fern-mdx-link svg {
  display: none;
}
`;

export function BadgeLinks({ badges }: { badges: BadgeItem[] }) {
  return (
    <div
      className="badge-links"
      style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}
    >
      <style dangerouslySetInnerHTML={{ __html: BADGE_LINKS_CSS }} />
      {badges.map((b) => (
        <a key={b.href} href={b.href} target="_blank" rel="noreferrer">
          <img src={b.src} alt={b.alt} />
        </a>
      ))}
    </div>
  );
}
