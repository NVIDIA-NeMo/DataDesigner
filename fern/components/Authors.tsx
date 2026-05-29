/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Authors - Renders author byline with avatars for dev notes / blog posts.
 *
 * Uses authors data from components/devnotes/authors-data.ts (synced with .authors.yml).
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 *
 * Usage in MDX (authors from frontmatter):
 *   ---
 *   authors:
 *     - jdoe
 *     - asmith
 *   ---
 *
 *   import { Authors } from "@/components/Authors";
 *   <Authors ids={authors} />
 */

import { authors } from "./devnotes/authors-data";

/**
 * Byline styles, injected by the component rather than loaded via docs.yml `css:`.
 * `css` is theme-owned, so under `global-theme: nvidia` a local `css:` list is
 * dropped at publish — styling has to ship with the component. See fern/docs.yml.
 */
const AUTHORS_CSS = `
.devnote-authors {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0 1.5rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(128, 128, 128, 0.15);
}
.devnote-authors__item {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.devnote-authors__avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  object-fit: cover;
}
.devnote-authors__meta {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}
.devnote-authors__name {
  font-weight: 600;
  font-size: 0.9rem;
}
.devnote-authors__description {
  font-size: 0.8rem;
  opacity: 0.75;
}
`;

export interface AuthorsProps {
  /** Author IDs from .authors.yml. From frontmatter: ids={authors} */
  ids?: string[];
}

export const Authors = ({ ids }: AuthorsProps) => {
  const validAuthors = (ids ?? [])
    .map((id) => authors[id])
    .filter(Boolean);

  if (validAuthors.length === 0) return null;

  return (
    <div className="devnote-authors">
      <style dangerouslySetInnerHTML={{ __html: AUTHORS_CSS }} />
      {validAuthors.map((author, i) => (
        <div key={i} className="devnote-authors__item">
          <img
            className="devnote-authors__avatar"
            src={author.avatar}
            alt=""
            width={32}
            height={32}
          />
          <div className="devnote-authors__meta">
            <span className="devnote-authors__name">{author.name}</span>
            <span className="devnote-authors__description">{author.description}</span>
          </div>
        </div>
      ))}
    </div>
  );
};
