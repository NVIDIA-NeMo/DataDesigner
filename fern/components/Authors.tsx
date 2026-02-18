/**
 * Authors - Renders author byline with avatars for dev notes.
 *
 * Uses authors data from components/devnotes/authors-data.ts (synced with .authors.yml).
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime.
 *
 * Usage in MDX (authors from frontmatter):
 *   ---
 *   authors:
 *     - dcorneil
 *     - etramel
 *   ---
 *
 *   import { Authors } from "@/components/Authors";
 *   <Authors ids={authors} />
 */

import { authors } from "./devnotes/authors-data";

export interface AuthorsProps {
  /** Author IDs from .authors.yml (e.g. dcorneil, etramel, kthadaka, nvidia). From frontmatter: ids={authors} */
  ids?: string[];
}

export const Authors = ({ ids }: AuthorsProps) => {
  const validAuthors = (ids ?? [])
    .map((id) => authors[id])
    .filter(Boolean);

  if (validAuthors.length === 0) return null;

  return (
    <div className="devnote-authors">
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
