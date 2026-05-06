import type { TrajectoryViewerProps } from "@/components/TrajectoryViewer";

const trajectory: Omit<TrajectoryViewerProps, "defaultOpen"> = {
  "question": "Who was the mother of the person under whom the colonizer of the winner of a border war with Libya in the 1st century BC reached its greatest extent?",
  "referenceAnswer": "Marcia",
  "goldenPassageHint": "⭐ = golden passage (contains evidence for the answer)",
  "summary": "Example trajectory: 4-hop question, 31 turns, 49 tool calls",
  "turns": [
    {
      "turnIndex": 1,
      "calls": [
        {
          "fn": "search",
          "arg": "border war Libya 1st century BC"
        }
      ]
    },
    {
      "turnIndex": 2,
      "calls": [
        {
          "fn": "search",
          "arg": "ancient Egypt Libya war conflict 1st century BC"
        },
        {
          "fn": "search",
          "arg": "Cyrene Libya ancient war border"
        }
      ]
    },
    {
      "turnIndex": 3,
      "calls": [
        {
          "fn": "search",
          "arg": "Ptolemaic Egypt Cyrenaica war conflict"
        },
        {
          "fn": "search",
          "arg": "Egypt border war ancient 1st century BC winner"
        }
      ]
    },
    {
      "turnIndex": 4,
      "calls": [
        {
          "fn": "open",
          "arg": "md_001400"
        }
      ]
    },
    {
      "turnIndex": 5,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt Marmarica border war ancient Roman"
        },
        {
          "fn": "search",
          "arg": "Marmaric War Egypt Libya ancient"
        }
      ]
    },
    {
      "turnIndex": 6,
      "calls": [
        {
          "fn": "search",
          "arg": "Greek colony Libya border war Cyrene colonization"
        },
        {
          "fn": "search",
          "arg": "Thera Cyrene colony Greek colonization ancient"
        }
      ]
    },
    {
      "turnIndex": 7,
      "calls": [
        {
          "fn": "open",
          "arg": "md_000167"
        }
      ]
    },
    {
      "turnIndex": 8,
      "calls": [
        {
          "fn": "search",
          "arg": "1st century BC war border Libya Egypt conflict"
        },
        {
          "fn": "search",
          "arg": "Chad Libya border war Aouzou Strip conflict"
        }
      ]
    },
    {
      "turnIndex": 9,
      "calls": [
        {
          "fn": "open",
          "arg": "md_001676 ⭐"
        }
      ]
    },
    {
      "turnIndex": 10,
      "calls": [
        {
          "fn": "search",
          "arg": "Roman Empire war Libya Numidia Africa ancient"
        },
        {
          "fn": "search",
          "arg": "Jugurthine War Numidia Rome Libya"
        }
      ]
    },
    {
      "turnIndex": 11,
      "calls": [
        {
          "fn": "open",
          "arg": "fw_021746"
        }
      ]
    },
    {
      "turnIndex": 12,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt ancient war tribes Libya Marmaridae border conflict"
        },
        {
          "fn": "search",
          "arg": "Cornelius Gallus Egypt prefect war border"
        }
      ]
    },
    {
      "turnIndex": 13,
      "calls": [
        {
          "fn": "open",
          "arg": "md_001054 ⭐"
        }
      ]
    },
    {
      "turnIndex": 14,
      "calls": [
        {
          "fn": "search",
          "arg": "Roman Empire greatest extent Trajan mother"
        },
        {
          "fn": "search",
          "arg": "Trajan Roman Empire greatest territorial extent"
        }
      ]
    },
    {
      "turnIndex": 15,
      "calls": [
        {
          "fn": "open",
          "arg": "md_002075 ⭐"
        },
        {
          "fn": "open",
          "arg": "md_002510 ⭐"
        }
      ]
    },
    {
      "turnIndex": 16,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt Roman province colony Augustus annexed"
        },
        {
          "fn": "search",
          "arg": "Trajan mother Marcia family"
        }
      ]
    },
    {
      "turnIndex": 17,
      "calls": [
        {
          "fn": "open",
          "arg": "md_002117"
        },
        {
          "fn": "search",
          "arg": "Trajan mother Marcia parents father"
        }
      ]
    },
    {
      "turnIndex": 18,
      "calls": [
        {
          "fn": "search",
          "arg": "Ulpia Marciana Trajan sister mother family"
        },
        {
          "fn": "search",
          "arg": "Egypt war Libya Marmaricae ancient Roman"
        }
      ]
    },
    {
      "turnIndex": 19,
      "calls": [
        {
          "fn": "search",
          "arg": "Marcia Trajan Roman empress mother emperor"
        },
        {
          "fn": "search",
          "arg": "Egypt Marmaric war ancient Rome Cyrene 1st century"
        }
      ]
    },
    {
      "turnIndex": 20,
      "calls": [
        {
          "fn": "open",
          "arg": "md_000722"
        }
      ]
    },
    {
      "turnIndex": 21,
      "calls": [
        {
          "fn": "search",
          "arg": "Ptolemaic Egypt border war Cyrene Cyrenaica territory"
        },
        {
          "fn": "search",
          "arg": "Egypt vs Libya ancient war Ptolemy Rome colony"
        }
      ]
    },
    {
      "turnIndex": 22,
      "calls": [
        {
          "fn": "search",
          "arg": "Toyota War Chad Libya border winner conflict"
        },
        {
          "fn": "search",
          "arg": "Chad Aouzou Strip war Libya France colonized"
        }
      ]
    },
    {
      "turnIndex": 23,
      "calls": [
        {
          "fn": "search",
          "arg": "1st century BC Egypt border war defeat victory Rome colonize"
        },
        {
          "fn": "search",
          "arg": "ancient Egypt war won Libya 1st century BC Rome province"
        }
      ]
    },
    {
      "turnIndex": 24,
      "calls": [
        {
          "fn": "open",
          "arg": "md_002076 ⭐"
        }
      ]
    },
    {
      "turnIndex": 25,
      "calls": [
        {
          "fn": "search",
          "arg": "French Empire greatest extent Napoleon mother"
        },
        {
          "fn": "search",
          "arg": "France Vichy border war Libya Chad colonial"
        }
      ]
    },
    {
      "turnIndex": 26,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt war 25 BC 30 BC Marmaridae tribe Roman conquest"
        },
        {
          "fn": "search",
          "arg": "Cornelius Balbus Garamantes Libya war Roman conquest"
        }
      ]
    },
    {
      "turnIndex": 27,
      "calls": [
        {
          "fn": "search",
          "arg": "ancient Crete war Libya border Greek colonize"
        },
        {
          "fn": "search",
          "arg": "Cyrene founded colonized by whom mother city Thera"
        }
      ]
    },
    {
      "turnIndex": 28,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt Roman province colonized 1st century BC Trajan mother Marcia"
        }
      ]
    },
    {
      "turnIndex": 29,
      "calls": [
        {
          "fn": "search",
          "arg": "Marcia Trajan mother Roman emperor name Ulpius"
        }
      ]
    },
    {
      "turnIndex": 30,
      "calls": [
        {
          "fn": "search",
          "arg": "Egypt war border Libya Cyrenaica defeated 1st century BC Roman province"
        }
      ]
    },
    {
      "turnIndex": 31,
      "calls": [
        {
          "fn": "answer",
          "body": "Based on my extensive research, I have gathered the following evidence:<br /><br />1. <strong>The Roman Empire reached its greatest territorial extent under Trajan</strong> (117 CE) — confirmed by multiple documents<br />2. <strong>Trajan's mother was from the Gens Marcia</strong> — the Trajan document states he \"on his mother's side from the Gens Marcia, of an Italic family of Sabine origin\"<br />3. <strong>Egypt was colonized/annexed by Rome</strong> after Cleopatra's death (30 BC), making it a Roman province<br />4. <strong>Egypt had border conflicts with Libya (ancient Marmaricae/Cyrenaica region)</strong> in the 1st century BC<br /><br />The answer is <strong>Marcia</strong>."
        }
      ]
    }
  ]
};

export default trajectory;
