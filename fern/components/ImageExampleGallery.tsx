/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Children, cloneElement, isValidElement, useState } from "react";
import type { ReactNode, SyntheticEvent } from "react";

import type { ImageExampleControlGroup } from "./ImageExample";

/**
 * ImageExampleGallery - compact inline gallery for generated image examples.
 *
 * Keeps one full-size example and its sampler controls visible while the rest
 * of the examples remain available through keyboard-accessible thumbnails.
 */

const IMAGE_EXAMPLE_GALLERY_CSS = `
.image-example-gallery {
  margin: 1.5rem 0 2.25rem;
  padding: 0.75rem;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  border-radius: 8px;
  background: var(--grayscale-1, rgba(128, 128, 128, 0.025));
}
.image-example-gallery__thumbs {
  display: flex;
  gap: 0.55rem;
  margin: 0 0 0.85rem;
  padding: 0 0 0.25rem;
  overflow-x: auto;
  overscroll-behavior-x: contain;
  scrollbar-width: thin;
}
.image-example-gallery__thumb {
  flex: 0 0 8.25rem;
  min-width: 0;
  padding: 0.28rem;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  border-radius: 8px;
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
  color: inherit;
  text-align: left;
  cursor: pointer;
}
.image-example-gallery__thumb:hover,
.image-example-gallery__thumb[aria-pressed="true"] {
  border-color: var(--accent, #76b900);
}
.image-example-gallery__thumb[aria-pressed="true"] {
  box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.22);
}
.image-example-gallery__thumb:focus-visible {
  outline: 3px solid var(--accent, #76b900);
  outline-offset: 2px;
}
.image-example-gallery__thumb-image,
.image-example-gallery__thumb-image-shell {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 10;
  margin: 0 !important;
  border-radius: 6px;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
}
.image-example-gallery__thumb-image {
  object-fit: cover;
}
.image-example-gallery__thumb img,
.image-example-gallery__thumb-image-shell,
.image-example-gallery__thumb-image-shell * {
  cursor: pointer !important;
}
.image-example-gallery__thumb-image-shell {
  overflow: hidden;
}
.image-example-gallery__thumb-image-shell > *,
.image-example-gallery__thumb-image-shell > * > * {
  display: block;
  width: 100%;
  height: 100%;
  margin: 0 !important;
  padding: 0 !important;
}
.image-example-gallery__thumb-image-shell img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  margin: 0 !important;
}
.image-example-gallery__thumb-title {
  display: block;
  margin-top: 0.34rem;
  font-size: 0.72rem;
  line-height: 1.2;
  font-weight: 650;
}
.image-example-gallery__detail {
  display: block;
}
.image-example-gallery__figure {
  margin: 0 !important;
  padding: 0;
}
.image-example-gallery__image-link {
  display: block;
  margin: 0 !important;
  border-radius: 8px;
  outline-offset: 3px;
  cursor: zoom-in;
}
.image-example-gallery__image-link:hover .image-example-gallery__image,
.image-example-gallery__image-link:hover .image-example-gallery__image-shell {
  border-color: var(--accent, #76b900);
  box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.24);
}
.image-example-gallery__image,
.image-example-gallery__image-shell {
  display: block;
  width: 100%;
  height: auto;
  max-height: 640px;
  object-fit: contain;
  margin: 0 !important;
  border-radius: 8px;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
}
.image-example-gallery__image-shell {
  overflow: hidden;
}
.image-example-gallery__image-shell > *,
.image-example-gallery__image-shell > * > * {
  display: block;
  width: 100%;
  margin: 0 !important;
  padding: 0 !important;
}
.image-example-gallery__image-shell img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 640px;
  object-fit: contain;
  margin: 0 !important;
}
.image-example-gallery__caption {
  margin-top: 0.55rem;
  font-size: 0.92rem;
  line-height: 1.35;
  font-weight: 650;
}
.image-example-gallery__control-groups {
  display: grid;
  gap: 0.65rem;
  margin-top: 0.85rem;
}
.image-example-gallery__group {
  border-left: 3px solid var(--accent, #76b900);
  padding-left: 0.65rem;
}
.image-example-gallery__group-label {
  display: block;
  margin: 0 0 0.3rem;
  color: var(--accent, #76b900);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  line-height: 1.2;
  text-transform: uppercase;
}
.image-example-gallery__chips {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.25rem;
  margin: 0;
  padding: 0;
  list-style: none;
}
.image-example-gallery__chip {
  min-width: 0;
  margin: 0 !important;
  padding: 0.34rem 0.45rem;
  border: 1px solid var(--grayscale-a5, rgba(128, 128, 128, 0.18));
  border-radius: 8px;
  background: var(--grayscale-2, rgba(128, 128, 128, 0.04));
  color: inherit;
  line-height: 1.22;
}
.image-example-gallery__key {
  display: block;
  margin: 0 0 0.12rem;
  color: var(--accent, #76b900);
  font-size: 0.61rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  line-height: 1.2;
  text-transform: uppercase;
}
.image-example-gallery__value {
  display: block;
  font-size: 0.76rem;
  overflow-wrap: anywhere;
}
@media (max-width: 840px) {
  .image-example-gallery {
    padding: 0.55rem;
  }
  .image-example-gallery__thumb {
    flex-basis: 7.35rem;
  }
}
`;

export interface ImageExampleGalleryItem {
  title: string;
  src: string;
  thumbnailSrc?: string;
  alt: string;
  controls?: [string, string][];
  controlGroups?: ImageExampleControlGroup[];
}

export interface ImageExampleGalleryProps {
  examples: ImageExampleGalleryItem[];
  defaultIndex?: number;
  children?: ReactNode;
}

interface ImageNodeProps {
  alt?: string;
  className?: string;
  decoding?: "async" | "auto" | "sync";
  loading?: "eager" | "lazy";
  src?: string;
}

const BASEPATH = "/nemo/datadesigner";

function withBasepath(path: string): string {
  if (!path.startsWith("/") || path.startsWith("//")) return path;
  if (path === BASEPATH || path.startsWith(`${BASEPATH}/`)) return path;
  return `${BASEPATH}${path}`;
}

function localPreviewAssetSrc(assetSrc: string): string | undefined {
  if (typeof document === "undefined" || !assetSrc.startsWith("/assets/")) {
    return undefined;
  }

  const localAsset = document.querySelector<HTMLImageElement>(
    'img[src^="/_local/"][src*="/fern/assets/"]'
  );
  const localAssetSrc = localAsset?.getAttribute("src");
  const fernAssetsIndex = localAssetSrc?.indexOf("/fern/assets/") ?? -1;

  if (!localAssetSrc || fernAssetsIndex < 0) return undefined;

  return `${localAssetSrc.slice(0, fernAssetsIndex + "/fern".length)}${assetSrc}`;
}

function handleImageError(event: SyntheticEvent<HTMLImageElement>) {
  const image = event.currentTarget;
  const fallbackSrc = image.dataset.fallbackSrc;

  if (!fallbackSrc) return;

  const resolvedFallbackSrc = localPreviewAssetSrc(fallbackSrc) ?? fallbackSrc;
  const link = image.closest("a.image-example-gallery__image-link");

  delete image.dataset.fallbackSrc;
  image.src = resolvedFallbackSrc;

  if (link instanceof HTMLAnchorElement) {
    link.href = resolvedFallbackSrc;
  }
}

function imageFallbackProps(primarySrc: string, fallbackSrc: string) {
  return primarySrc !== fallbackSrc
    ? { "data-fallback-src": fallbackSrc, onError: handleImageError }
    : {};
}

function exampleLabel(title: string): string {
  return title.split(":")[0] ?? title;
}

function mergeClassNames(...classNames: (string | undefined)[]): string {
  return classNames.filter(Boolean).join(" ");
}

function imageNodeSrc(imageNode: ReactNode): string | undefined {
  if (!isValidElement<ImageNodeProps>(imageNode)) return undefined;
  return typeof imageNode.props.src === "string" ? imageNode.props.src : undefined;
}

function renderProvidedImage(
  imageNode: ReactNode,
  shellClassName: string,
  imageClassName: string,
  alt: string
) {
  const content = isValidElement<ImageNodeProps>(imageNode)
    ? cloneElement(imageNode, {
        alt,
        className: mergeClassNames(imageNode.props.className, imageClassName),
        decoding: imageNode.props.decoding ?? "async",
        loading: imageNode.props.loading ?? "lazy",
      })
    : imageNode;

  return <span className={shellClassName}>{content}</span>;
}

function renderGroup(group: ImageExampleControlGroup, groupIndex: number) {
  return (
    <div className="image-example-gallery__group" key={groupIndex}>
      <span className="image-example-gallery__group-label">{group.label}</span>
      <ul className="image-example-gallery__chips">
        {group.controls.map(([key, value], controlIndex) => (
          <li className="image-example-gallery__chip" key={`${groupIndex}-${controlIndex}`}>
            <span className="image-example-gallery__key">{key}</span>
            <span className="image-example-gallery__value">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const ImageExampleGallery = ({
  examples,
  defaultIndex = 0,
  children,
}: ImageExampleGalleryProps) => {
  const initialIndex = Math.min(Math.max(defaultIndex, 0), Math.max(examples.length - 1, 0));
  const [selectedIndex, setSelectedIndex] = useState(initialIndex);
  const selectedExample = examples[selectedIndex];
  const imageNodes = Children.toArray(children).filter((child) => isValidElement(child));

  if (!selectedExample) return null;

  const groups =
    selectedExample.controlGroups && selectedExample.controlGroups.length > 0
      ? selectedExample.controlGroups
      : [{ label: "Sampler controls", controls: selectedExample.controls ?? [] }];
  const selectedImageNode = imageNodes[selectedIndex];
  const fullSrc = imageNodeSrc(selectedImageNode) ?? withBasepath(selectedExample.src);
  const fullFallbackProps = imageFallbackProps(fullSrc, selectedExample.src);

  return (
    <div className="image-example-gallery">
      {/* static CSS string literal (no user input) — safe to inject as raw HTML */}
      <style dangerouslySetInnerHTML={{ __html: IMAGE_EXAMPLE_GALLERY_CSS }} />
      <div className="image-example-gallery__thumbs" role="list" aria-label="Image examples">
        {examples.map((example, index) => {
          const rawThumbSrc = example.thumbnailSrc ?? example.src;
          const thumbSrc = withBasepath(rawThumbSrc);
          const thumbFallbackProps = imageFallbackProps(thumbSrc, rawThumbSrc);
          const thumbImageNode = imageNodes[index];

          return (
            <button
              aria-label={`Show ${example.title}`}
              aria-pressed={index === selectedIndex}
              className="image-example-gallery__thumb"
              key={`${example.src}-${index}`}
              onClickCapture={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setSelectedIndex(index);
              }}
              type="button"
            >
              {thumbImageNode ? (
                renderProvidedImage(
                  thumbImageNode,
                  "image-example-gallery__thumb-image-shell",
                  "image-example-gallery__thumb-image-node",
                  ""
                )
              ) : (
                <img
                  alt=""
                  className="image-example-gallery__thumb-image"
                  decoding="async"
                  loading="lazy"
                  src={thumbSrc}
                  {...thumbFallbackProps}
                />
              )}
              <span className="image-example-gallery__thumb-title">{exampleLabel(example.title)}</span>
            </button>
          );
        })}
      </div>
      <div className="image-example-gallery__detail">
        <figure className="image-example-gallery__figure">
          <a
            aria-label={`Open full-size image: ${selectedExample.title}`}
            className="image-example-gallery__image-link"
            href={fullSrc}
            onClickCapture={(event) => event.preventDefault()}
            rel="noopener noreferrer"
            target="_blank"
          >
            {selectedImageNode ? (
              renderProvidedImage(
                selectedImageNode,
                "image-example-gallery__image-shell",
                "image-example-gallery__image-node",
                selectedExample.alt
              )
            ) : (
              <img
                alt={selectedExample.alt}
                className="image-example-gallery__image"
                decoding="async"
                loading="lazy"
                src={fullSrc}
                {...fullFallbackProps}
              />
            )}
          </a>
          <figcaption className="image-example-gallery__caption">{selectedExample.title}</figcaption>
        </figure>
        <div className="image-example-gallery__control-groups">{groups.map(renderGroup)}</div>
      </div>
    </div>
  );
};
