/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Figure - Centered image with an optional NVIDIA-green italic caption.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 *
 * Usage in MDX:
 *   import { Figure } from "@/components/Figure";
 *
 *   <Figure src="/assets/foo.png" alt="..." width={600}>
 *     Caption text with **inline markdown** if helpful.
 *   </Figure>
 */

/**
 * Figure styles, injected by the component rather than loaded via docs.yml `css:`.
 * `css` is theme-owned, so under `global-theme: nvidia` a local `css:` list is
 * dropped at publish — styling has to ship with the component. See fern/docs.yml
 * and the same pattern in Authors.tsx.
 */
const FIGURE_CSS = `
.devnote-figure {
  text-align: center;
  margin: 1.25rem 0;
}
.devnote-figure__img {
  max-width: 100%;
  height: auto;
}
.devnote-figure__caption {
  display: block;
  margin-top: 0.5rem;
  color: #76B900;
  font-size: 0.85em;
  font-style: italic;
  line-height: 1.4;
}
`;

export interface FigureProps {
  /** Image source path (e.g. "/assets/<slug>/foo.png"). */
  src: string;
  /** Alt text for accessibility. */
  alt: string;
  /** Optional explicit width in pixels (or any CSS length). */
  width?: number | string;
  /** Caption content; rendered only if children are provided. */
  children?: React.ReactNode;
}

export const Figure = ({ src, alt, width, children }: FigureProps) => (
  <div className="devnote-figure">
    {/* static CSS string literal (no user input) — safe to inject as raw HTML */}
    <style dangerouslySetInnerHTML={{ __html: FIGURE_CSS }} />
    <img className="devnote-figure__img" src={src} alt={alt} width={width} />
    {children && <em className="devnote-figure__caption">{children}</em>}
  </div>
);
