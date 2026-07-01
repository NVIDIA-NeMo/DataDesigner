#let data = json("results.json")
#let quality = data.at("quality").at("routes")
#let diversity = data.at("diversity").at("routes")
#let weak = quality.at("dd-weak")
#let strong = quality.at("dd-strong")
#let hinted = quality.at("dd-hinted")
#let weak-diversity = diversity.at("dd-weak")
#let mixed-diversity = diversity.at("dd-mixed")
#let cost-reduction = 1 - hinted.at("estimated_cost_usd") / strong.at("estimated_cost_usd")

#let pct(value) = str(calc.round(value * 1000) / 10) + "%"
#let money(value) = "$" + str(calc.round(value * 1000000) / 1000000)
#let number(value, digits: 2) = str(calc.round(value * calc.pow(10, digits)) / calc.pow(10, digits))

#set page(
  paper: "a4",
  margin: (x: 18mm, y: 17mm),
  numbering: "1 / 1",
)
#set text(font: "New Computer Modern", size: 9.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.6em)[
  #set text(size: 15pt, weight: "bold", fill: rgb("#2d3a50"))
  #it.body
]
#show heading.where(level: 2): it => block(above: 1em, below: 0.4em)[
  #set text(size: 11pt, weight: "bold", fill: rgb("#3f5f8f"))
  #it.body
]

#align(center)[
  #text(size: 22pt, weight: "bold", fill: rgb("#20324d"))[Data Designer + Switchyard]
  #v(3pt)
  #text(size: 13pt, fill: rgb("#4d6585"))[Measured routing experiments]
  #v(6pt)
  #text(size: 8.5pt, fill: gray)[Generated #data.at("generated_at")]
]

#v(8pt)
#block(fill: rgb("#edf3fa"), inset: 10pt, radius: 5pt)[
  *Result.* Task-hinted routing achieved #pct(hinted.at("accuracy")) exact-answer accuracy at an estimated
  cost of #money(hinted.at("estimated_cost_usd")), compared with #pct(strong.at("accuracy")) at
  #money(strong.at("estimated_cost_usd")) for strong-only. This is a #pct(cost-reduction) cost reduction.
  Weak-only achieved #pct(weak.at("accuracy")) at
  #money(weak.at("estimated_cost_usd")) for weak-only.
]

= Experiment design

The benchmark keeps Data Designer's schema, seeding, structured-output handling, and async execution unchanged. A single OpenAI-compatible provider points to five logical Switchyard routes:

- `dd-weak`: direct weak model
- `dd-strong`: direct strong model
- `dd-hinted`: Data Designer supplies an easy/hard hint and Switchyard selects the matching tier
- `dd-smart`: optional LLM-classifier route used during calibration
- `dd-mixed`: seeded 35% strong / 65% weak routing for creative generation

The quality set contains #data.at("quality").at("task_count") exact-answer tasks: sixteen simple transformations or arithmetic tasks and four combinatorics or number-theory tasks. The diversity run generates #data.at("diversity").at("record_count") responses to one constrained support-ticket prompt.

== Models

#table(
  columns: (1fr, 2.4fr),
  inset: 5pt,
  stroke: rgb("#ccd7e5"),
  fill: (_, row) => if row == 0 { rgb("#dce7f5") } else { white },
  [*Role*], [*Model*],
  [Weak], [#data.at("models").at("weak_model")],
  [Strong], [#data.at("models").at("strong_model")],
  [Classifier], [#data.at("models").at("classifier_model")],
)

= Quality and estimated cost

#table(
  columns: (1.2fr, 0.75fr, 0.7fr, 0.7fr, 0.8fr, 0.75fr, 0.85fr),
  inset: 4pt,
  stroke: rgb("#ccd7e5"),
  fill: (_, row) => if row == 0 { rgb("#dce7f5") } else if calc.rem(row, 2) == 0 { rgb("#f7f9fc") } else { white },
  [*Route*], [*All*], [*Easy*], [*Hard*], [*S / W*], [*Classifier*], [*Est. cost*],
  [Weak], [#pct(weak.at("accuracy"))], [#pct(weak.at("easy_accuracy"))], [#pct(weak.at("hard_accuracy"))], [`- / -`], [0], [#money(weak.at("estimated_cost_usd"))],
  [Strong], [#pct(strong.at("accuracy"))], [#pct(strong.at("easy_accuracy"))], [#pct(strong.at("hard_accuracy"))], [`- / -`], [0], [#money(strong.at("estimated_cost_usd"))],
  [Hinted], [#pct(hinted.at("accuracy"))], [#pct(hinted.at("easy_accuracy"))], [#pct(hinted.at("hard_accuracy"))], [#hinted.at("strong_calls") / #hinted.at("weak_calls")], [#hinted.at("classifier_calls")], [#money(hinted.at("estimated_cost_usd"))],
)

#v(5pt)
Cost per correct row was #money(weak.at("cost_per_correct_row_usd")) for weak-only,
#money(strong.at("cost_per_correct_row_usd")) for strong-only, and
#money(hinted.at("cost_per_correct_row_usd")) for task-hinted routing.

= Diversity

#table(
  columns: (1.25fr, 0.85fr, 0.85fr, 0.9fr, 0.95fr, 0.85fr),
  inset: 4pt,
  stroke: rgb("#ccd7e5"),
  fill: (_, row) => if row == 0 { rgb("#dce7f5") } else if calc.rem(row, 2) == 0 { rgb("#f7f9fc") } else { white },
  [*Route*], [*Vocabulary*], [*Distinct-1*], [*Distinct-2*], [*Mean Jaccard*], [*Duplicates*],
  [Weak], [#weak-diversity.at("vocabulary_size")], [#number(weak-diversity.at("distinct_1"), digits: 3)], [#number(weak-diversity.at("distinct_2"), digits: 3)], [#number(weak-diversity.at("mean_pairwise_jaccard"), digits: 3)], [#pct(weak-diversity.at("exact_duplicate_rate"))],
  [Mixed], [#mixed-diversity.at("vocabulary_size")], [#number(mixed-diversity.at("distinct_1"), digits: 3)], [#number(mixed-diversity.at("distinct_2"), digits: 3)], [#number(mixed-diversity.at("mean_pairwise_jaccard"), digits: 3)], [#pct(mixed-diversity.at("exact_duplicate_rate"))],
)

Higher distinct-n scores and vocabulary are better. Lower pairwise Jaccard similarity and duplicate rate indicate more varied outputs. The mixed route increased vocabulary, but worsened both distinct-n and pairwise similarity. This small run does not support model mixing as a diversity improvement.

= Interpretation

The experiment tests Switchyard as an operational layer below Data Designer, not as a replacement for Data Designer model aliases. The useful addition is per-request backend selection, one logical endpoint, centralized usage accounting, and the ability to change the model mixture without changing dataset configuration. The prompt marker used here should become explicit request metadata in a production integration.

The benchmark is intentionally small. Before production use, repeat it with domain-specific validators, more samples, stable endpoint conditions, and a route policy trained or calibrated on representative prompts.
