/** Extracted from docs/devnotes/posts/deep-research-trajectories.md - 4-hop example trajectory */

import type { TrajectoryViewerProps } from "../TrajectoryViewer";

const turns = [
  { turnIndex: 1, calls: [{ fn: "search" as const, arg: '"border war Libya 1st century BC"' }] },
  {
    turnIndex: 2,
    calls: [
      { fn: "search" as const, arg: '"ancient Egypt Libya war conflict 1st century BC"' },
      { fn: "search" as const, arg: '"Cyrene Libya ancient war border"' },
    ],
  },
  {
    turnIndex: 3,
    calls: [
      { fn: "search" as const, arg: '"Ptolemaic Egypt Cyrenaica war conflict"' },
      { fn: "search" as const, arg: '"Egypt border war ancient 1st century BC winner"' },
    ],
  },
  { turnIndex: 4, calls: [{ fn: "open" as const, arg: "md_001400" }] },
  {
    turnIndex: 5,
    calls: [
      { fn: "search" as const, arg: '"Egypt Marmarica border war ancient Roman"' },
      { fn: "search" as const, arg: '"Marmaric War Egypt Libya ancient"' },
    ],
  },
  {
    turnIndex: 6,
    calls: [
      { fn: "search" as const, arg: '"Greek colony Libya border war Cyrene colonization"' },
      { fn: "search" as const, arg: '"Thera Cyrene colony Greek colonization ancient"' },
    ],
  },
  { turnIndex: 7, calls: [{ fn: "open" as const, arg: "md_000167" }] },
  {
    turnIndex: 8,
    calls: [
      { fn: "search" as const, arg: '"1st century BC war border Libya Egypt conflict"' },
      { fn: "search" as const, arg: '"Chad Libya border war Aouzou Strip conflict"' },
    ],
  },
  { turnIndex: 9, calls: [{ fn: "open" as const, arg: "md_001676", isGolden: true }] },
  {
    turnIndex: 10,
    calls: [
      { fn: "search" as const, arg: '"Roman Empire war Libya Numidia Africa ancient"' },
      { fn: "search" as const, arg: '"Jugurthine War Numidia Rome Libya"' },
    ],
  },
  { turnIndex: 11, calls: [{ fn: "open" as const, arg: "fw_021746" }] },
  {
    turnIndex: 12,
    calls: [
      { fn: "search" as const, arg: '"Egypt ancient war tribes Libya Marmaridae border conflict"' },
      { fn: "search" as const, arg: '"Cornelius Gallus Egypt prefect war border"' },
    ],
  },
  { turnIndex: 13, calls: [{ fn: "open" as const, arg: "md_001054", isGolden: true }] },
  {
    turnIndex: 14,
    calls: [
      { fn: "search" as const, arg: '"Roman Empire greatest extent Trajan mother"' },
      { fn: "search" as const, arg: '"Trajan Roman Empire greatest territorial extent"' },
    ],
  },
  {
    turnIndex: 15,
    calls: [
      { fn: "open" as const, arg: "md_002075", isGolden: true },
      { fn: "open" as const, arg: "md_002510", isGolden: true },
    ],
  },
  {
    turnIndex: 16,
    calls: [
      { fn: "search" as const, arg: '"Egypt Roman province colony Augustus annexed"' },
      { fn: "search" as const, arg: '"Trajan mother Marcia family"' },
    ],
  },
  {
    turnIndex: 17,
    calls: [
      { fn: "open" as const, arg: "md_002117" },
      { fn: "search" as const, arg: '"Trajan mother Marcia parents father"' },
    ],
  },
  {
    turnIndex: 18,
    calls: [
      { fn: "search" as const, arg: '"Ulpia Marciana Trajan sister mother family"' },
      { fn: "search" as const, arg: '"Egypt war Libya Marmaricae ancient Roman"' },
    ],
  },
  {
    turnIndex: 19,
    calls: [
      { fn: "search" as const, arg: '"Marcia Trajan Roman empress mother emperor"' },
      { fn: "search" as const, arg: '"Egypt Marmaric war ancient Rome Cyrene 1st century"' },
    ],
  },
  { turnIndex: 20, calls: [{ fn: "open" as const, arg: "md_000722" }] },
  {
    turnIndex: 21,
    calls: [
      { fn: "search" as const, arg: '"Ptolemaic Egypt border war Cyrene Cyrenaica territory"' },
      { fn: "search" as const, arg: '"Egypt vs Libya ancient war Ptolemy Rome colony"' },
    ],
  },
  {
    turnIndex: 22,
    calls: [
      { fn: "search" as const, arg: '"Toyota War Chad Libya border winner conflict"' },
      { fn: "search" as const, arg: '"Chad Aouzou Strip war Libya France colonized"' },
    ],
  },
  {
    turnIndex: 23,
    calls: [
      { fn: "search" as const, arg: '"1st century BC Egypt border war defeat victory Rome colonize"' },
      { fn: "search" as const, arg: '"ancient Egypt war won Libya 1st century BC Rome province"' },
    ],
  },
  { turnIndex: 24, calls: [{ fn: "open" as const, arg: "md_002076", isGolden: true }] },
  {
    turnIndex: 25,
    calls: [
      { fn: "search" as const, arg: '"French Empire greatest extent Napoleon mother"' },
      { fn: "search" as const, arg: '"France Vichy border war Libya Chad colonial"' },
    ],
  },
  {
    turnIndex: 26,
    calls: [
      { fn: "search" as const, arg: '"Egypt war 25 BC 30 BC Marmaridae tribe Roman conquest"' },
      { fn: "search" as const, arg: '"Cornelius Balbus Garamantes Libya war Roman conquest"' },
    ],
  },
  {
    turnIndex: 27,
    calls: [
      { fn: "search" as const, arg: '"ancient Crete war Libya border Greek colonize"' },
      { fn: "search" as const, arg: '"Cyrene founded colonized by whom mother city Thera"' },
    ],
  },
  {
    turnIndex: 28,
    calls: [
      {
        fn: "search" as const,
        arg: '"Egypt Roman province colonized 1st century BC Trajan mother Marcia"',
      },
    ],
  },
  {
    turnIndex: 29,
    calls: [{ fn: "search" as const, arg: '"Marcia Trajan mother Roman emperor name Ulpius"' }],
  },
  {
    turnIndex: 30,
    calls: [
      {
        fn: "search" as const,
        arg: '"Egypt war border Libya Cyrenaica defeated 1st century BC Roman province"',
      },
    ],
  },
  {
    turnIndex: 31,
    calls: [
      {
        fn: "answer" as const,
        body: `Based on my extensive research, I have gathered the following evidence:<br><br>1. <strong>The Roman Empire reached its greatest territorial extent under Trajan</strong> (117 CE) — confirmed by multiple documents<br>2. <strong>Trajan's mother was from the Gens Marcia</strong> — the Trajan document states he "on his mother's side from the Gens Marcia, of an Italic family of Sabine origin"<br>3. <strong>Egypt was colonized/annexed by Rome</strong> after Cleopatra's death (30 BC), making it a Roman province<br>4. <strong>Egypt had border conflicts with Libya (ancient Marmaricae/Cyrenaica region)</strong> in the 1st century BC<br><br>The answer is <strong>Marcia</strong>.`,
      },
    ],
  },
];

const trajectory: TrajectoryViewerProps = {
  question:
    "Who was the mother of the person under whom the colonizer of the winner of a border war with Libya in the 1st century BC reached its greatest extent?",
  referenceAnswer: "Marcia",
  goldenPassageHint: "⭐ = golden passage (contains evidence for the answer)",
  turns,
  summary: "Example trajectory: 4-hop question, 31 turns, 49 tool calls",
  defaultOpen: true,
};

export default trajectory;
