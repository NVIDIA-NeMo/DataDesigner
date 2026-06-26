/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ImageExample - Image-first figure for generated image examples.
 *
 * Designed for dev notes where an example image should stay visually dominant
 * while sampler controls remain scannable beside or below it.
 *
 * Usage in MDX:
 *   import { ImageExample } from "@/components/ImageExample";
 *
 *   <ImageExample
 *     title="Example 1: ..."
 *     src="/assets/foo/example.jpg"
 *     alt="..."
 *     controls={[
 *       ["document type", "invoice"],
 *       ["scan condition", "faded photocopy"],
 *     ]}
 *   />
 */

const IMAGE_EXAMPLE_CSS = `
.image-example {
  margin: 1.5rem 0 2.25rem;
  padding: 0.75rem;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  border-radius: 8px;
  background: var(--grayscale-1, rgba(128, 128, 128, 0.025));
}
.image-example__figure {
  margin: 0 !important;
  padding: 0;
}
.image-example__image-link {
  display: block;
  margin: 0 auto !important;
  padding: 0;
  border-radius: 8px;
  outline-offset: 3px;
  cursor: zoom-in;
}
.image-example__image-link:hover .image-example__image {
  border-color: var(--accent, #76b900);
  box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.24);
}
.image-example__image {
  display: block;
  width: 100%;
  height: auto;
  margin: 0 !important;
  border-radius: 8px;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
}
.image-example__caption {
  margin-top: 0.55rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--grayscale-a4, rgba(128, 128, 128, 0.14));
  font-size: 0.92rem;
  line-height: 1.35;
  font-weight: 650;
}
.image-example__control-groups {
  display: grid;
  gap: 0.55rem;
  margin-top: 0.6rem;
}
.image-example__control-groups--split {
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.image-example__group {
  border-left: 3px solid var(--accent, #76b900);
  padding-left: 0.65rem;
}
.image-example__group-label {
  display: block;
  margin: 0 0 0.3rem;
  color: var(--accent, #76b900);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  line-height: 1.2;
  text-transform: uppercase;
}
.image-example__chips {
  column-count: 2;
  column-gap: 0.36rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.image-example__control-groups--split .image-example__chips {
  column-count: 2;
}
.image-example__chip {
  display: inline-block;
  width: 100%;
  min-width: 0;
  margin: 0 0 0.2rem !important;
  padding: 0.34rem 0.45rem;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  border-radius: 8px;
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
  color: inherit;
  line-height: 1.22;
  break-inside: avoid;
}
.image-example__key {
  display: block;
  margin: 0 0 0.12rem;
  color: var(--accent, #76b900);
  font-size: 0.61rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  line-height: 1.2;
  text-transform: uppercase;
}
.image-example__value {
  display: block;
  font-size: 0.76rem;
  overflow-wrap: anywhere;
}
.image-example__lightbox {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 9999;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}
.image-example__lightbox:target {
  display: flex;
}
.image-example__lightbox-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.82);
}
.image-example__lightbox-panel {
  position: relative;
  z-index: 1;
  max-width: min(94vw, 1200px);
  max-height: 90vh;
}
.image-example__lightbox-image {
  display: block;
  max-width: 100%;
  max-height: 90vh;
  width: auto;
  height: auto;
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
}
.image-example__lightbox-close {
  position: absolute;
  top: -0.85rem;
  right: -0.85rem;
  display: flex;
  width: 2rem;
  height: 2rem;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #fff;
  color: #111;
  font-size: 1.25rem;
  line-height: 1;
  text-decoration: none !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.32);
}
@media (max-width: 720px) {
  .image-example {
    padding: 0.55rem;
  }
  .image-example__chips,
  .image-example__control-groups--split .image-example__chips {
    column-count: 1;
  }
  .image-example__image-link {
    max-width: 100% !important;
  }
  .image-example__lightbox {
    padding: 1rem;
  }
}
`;

export interface ImageExampleControlGroup {
  label: string;
  controls: [string, string][];
}

export interface ImageExampleProps {
  title: string;
  src: string;
  alt: string;
  imageWidth?: string;
  controls?: [string, string][];
  controlGroups?: ImageExampleControlGroup[];
}

const BASEPATH = "/nemo/datadesigner";

function withBasepath(path: string): string {
  if (!path.startsWith("/") || path.startsWith("//")) return path;
  if (path === BASEPATH || path.startsWith(`${BASEPATH}/`)) return path;
  return `${BASEPATH}${path}`;
}

function handleImageError(event: React.SyntheticEvent<HTMLImageElement>) {
  const image = event.currentTarget;
  const fallbackSrc = image.dataset.fallbackSrc;

  if (!fallbackSrc) return;

  delete image.dataset.fallbackSrc;
  image.src = fallbackSrc;
}

function lightboxIdFor(src: string): string {
  const slug = src.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return `image-example-${slug.slice(-96)}`;
}

function renderGroup(group: ImageExampleControlGroup, groupIndex: number) {
  return (
    <div className="image-example__group" key={groupIndex}>
      <span className="image-example__group-label">{group.label}</span>
      <ul className="image-example__chips">
        {group.controls.map(([key, value], controlIndex) => (
          <li className="image-example__chip" key={`${groupIndex}-${controlIndex}`}>
            <span className="image-example__key">{key}</span>
            <span className="image-example__value">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const ImageExample = ({
  title,
  src,
  alt,
  imageWidth,
  controls = [],
  controlGroups,
}: ImageExampleProps) => {
  const groups =
    controlGroups && controlGroups.length > 0
      ? controlGroups
      : [{ label: "Sampler controls", controls }];
  const fallbackSrc = withBasepath(src);
  const hasFallback = fallbackSrc !== src;
  const lightboxId = lightboxIdFor(src);
  const imageFallbackProps = hasFallback
    ? { "data-fallback-src": fallbackSrc, onError: handleImageError }
    : {};

  return (
    <div className="image-example">
      {/* static CSS string literal (no user input) — safe to inject as raw HTML */}
      <style dangerouslySetInnerHTML={{ __html: IMAGE_EXAMPLE_CSS }} />
      <figure className="image-example__figure">
        <a
          className="image-example__image-link"
          href={`#${lightboxId}`}
          style={imageWidth ? { maxWidth: imageWidth } : undefined}
          aria-label={`Open image preview: ${title}`}
        >
          <img className="image-example__image" src={src} alt={alt} {...imageFallbackProps} />
        </a>
        <div
          className="image-example__lightbox"
          id={lightboxId}
          role="dialog"
          aria-modal="true"
          aria-label={title}
        >
          <a
            className="image-example__lightbox-backdrop"
            href="#_"
            aria-label="Close image preview"
          />
          <div className="image-example__lightbox-panel">
            <a className="image-example__lightbox-close" href="#_" aria-label="Close image preview">
              &times;
            </a>
            <img
              className="image-example__lightbox-image"
              src={src}
              alt={alt}
              {...imageFallbackProps}
            />
          </div>
        </div>
        <figcaption className="image-example__caption">{title}</figcaption>
      </figure>
      <div
        className={
          groups.length > 1
            ? "image-example__control-groups image-example__control-groups--split"
            : "image-example__control-groups"
        }
      >
        {groups.map(renderGroup)}
      </div>
    </div>
  );
};
