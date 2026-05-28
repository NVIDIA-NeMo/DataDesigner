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

export function BadgeLinks({ badges }: { badges: BadgeItem[] }) {
  return (
    <div
      className="badge-links"
      style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}
    >
      {badges.map((b) => (
        <a key={b.href} href={b.href} target="_blank" rel="noreferrer">
          <img src={b.src} alt={b.alt} />
        </a>
      ))}
    </div>
  );
}
