# Preliminary results: does retrieval help an LLM screen papers?

LitScreen screens papers for a systematic review: for each title and abstract
it decides **include**, **exclude** or **unsure** against the review's written
criteria. This page compares three setups on real screening decisions from the
[SYNERGY](https://github.com/asreview/synergy-dataset) dataset:

- **RAG**: the prompt contains the criteria plus the 3 already-screened papers
  most similar to the candidate (`src/screening_chain.py`).
- **Baseline**: the same prompt with no retrieved examples (`--k-examples 0`).
- **Agent**: the model decides when and what to search, and can flag a paper
  for human review, which makes its decision unsure
  (`src/screening_agent_tools.py`).

20 runs across two reviews, as of 5 October 2026. These results are
preliminary: see [Limits](#limits).

## Short answer

Retrieval makes the model more decisive without making it less safe. On the
Xu_2022 review it raised strict recall from about 0.37 to 0.44 and left 11–15%
fewer papers for a person to check, on both data splits. It did not change
precision, and no run of any setup excluded a paper that ended up in the final
review.

The tool-using agent did not improve on this. It was the most cautious setup,
leaving about 24 more papers for a person than the fixed chain, with the same
precision. Retrieval helps the screening workflow by reducing manual work; it
does not make borderline judgements more accurate.

## Findings

| Finding | Evidence |
|---|---|
| **Higher strict recall with retrieval** (repeated) | 0.37 → 0.44 on split 1 and 0.38 → 0.43 on split 2. Every RAG run beat every baseline run on both splits. |
| **Fewer papers left for a person** (repeated) | Unsure decisions dropped from 97 to 86 (split 1) and from 88 to 75 (split 2) out of 276, about 4–5 percentage points at the review's real inclusion rate. Ranges overlap on split 1, so the size is less certain than the direction. |
| **Precision unchanged** (no effect) | Corrected precision 0.23 vs 0.21 (split 1) and 0.28 vs 0.28 (split 2). When retrieval turns an unsure into an include, it is right about half the time. |
| **No relevant papers lost** (all setups) | No run excluded a paper the reviewers kept after reading the full text (20 such papers in split 1's test set, 17 in split 2's). |
| **Agent: more cautious, not more accurate** | On split 2 (2 runs): strict recall 0.32 vs 0.43 for RAG, 99 vs 75 unsure papers, precision 0.30 vs 0.28. It searched 1.02 times per paper on average, and all its unsure decisions came from flagging. It needs 2–3 model calls per paper instead of 1. |
| **Criteria wording matters more than retrieval** (large effect) | On Donners_2021, removing one inclusion clause ("access to the full text in English") dropped precision from 1.00 to about 0.55, because conference abstracts started being included. |
| **The model screens like a strict final reviewer** (observed) | Against the reviewers' lenient title/abstract decisions, strict recall looks low. The "missed" papers are mostly ones the reviewers dropped themselves after reading the full text, often studies the written criteria explicitly exclude. |

## Xu_2022: international students' learning environment

Review: Xu et al., *A conducive learning environment in international higher
education*, Educational Research Review (2022),
[doi:10.1016/j.edurev.2022.100474](https://doi.org/10.1016/j.edurev.2022.100474).
Judgement-heavy criteria: 4 inclusion and 8 exclusion rules. Scored against the
reviewers' title/abstract decisions. Mean of 3 runs (agent: 2), range in
brackets.

| Setup | Recall (strict) | Unsure (of 276) | Unsure % (adj) | Precision (adj) | Recall (unsure kept) | Final lost |
|---|---|---|---|---|---|---|
| **Split 1** (random state 42) | | | | | | |
| RAG | 0.44 (0.43–0.45) | 86 (81–92) | 29.0 (26.3–31.4) | 0.23 (0.21–0.24) | 0.84 (0.82–0.86) | 0 of 20 |
| Baseline | 0.37 (0.37–0.38) | 97 (90–102) | 33.1 (30.5–35.3) | 0.21 (0.20–0.23) | 0.80 (0.78–0.82) | 0 of 20 |
| **Split 2** (random state 7) | | | | | | |
| RAG | 0.43 (0.41–0.49) | 75 (74–76) | 23.9 (23.7–24.2) | 0.28 (0.25–0.30) | 0.82 (0.79–0.86) | 0 of 17 |
| Baseline | 0.38 (0.36–0.39) | 88 (87–89) | 28.8 (28.4–29.0) | 0.28 (0.27–0.30) | 0.82 (0.80–0.83) | 0 of 17 |
| Agent | 0.32 (0.32–0.32) | 99 (98–100) | 32.1 (31.9–32.4) | 0.30 (0.27–0.33) | 0.82 (0.79–0.84) | 0 of 17 |

Raw precision on the downsampled test set ranged from 0.50 to 0.67 across runs.
The (adj) columns correct for keeping 200 of 824 available excludes: each test
exclude stands for 4.12 real ones. No paper failed in any run.

## Donners_2021: emicizumab pharmacokinetics

Review: Donners et al., *Pharmacokinetics and Associated Efficacy of
Emicizumab in Humans: A Systematic Review*, Clinical Pharmacokinetics (2021),
[doi:10.1007/s40262-021-01042-w](https://doi.org/10.1007/s40262-021-01042-w).
Specific criteria (human data, original or modelled PK data, full text in
English). 84 test papers, 9 includes, scored against final decisions.

| Setup | Run | Precision | Recall (strict) | Unsure (of 84) | Final lost |
|---|---|---|---|---|---|
| **Original criteria** | | | | | |
| RAG | 1* | 1.00 | 0.89 | 13 | 0 of 9 |
| RAG | 2 | 1.00 | 0.89 | 11 | 0 of 9 |
| Baseline | 1* | 1.00 | 0.89 | 15 | 0 of 9 |
| Baseline | 2 | 0.80 | 0.89 | 14 | 0 of 9 |
| **Full-text clause removed** (sensitivity check) | | | | | |
| RAG | 1* | 0.53 | 0.89 | 4 | 1 of 9 |
| RAG | 2 | 0.57 | 0.89 | 5 | 0 of 9 |

\* Summary numbers only: these runs' result files were overwritten before
per-run tagging existed. The one include missed in every run has no PK data in
its abstract; the reviewers likely judged it from the full text.

## How the evaluation works

- **Data**: SYNERGY+ v3.0 ([doi:10.34894/DDCVCV](https://doi.org/10.34894/DDCVCV)),
  which exports only open-access records with an abstract of at least 20 words
  or 100 characters. Xu_2022: 960 of 3,165 records (86 title/abstract includes,
  21 final includes). Donners_2021: 119 of 260 records (14 final includes).
  Abstracts are not included in this repository, in line with the SYNERGY
  licence; only aggregate numbers are reported here.
- **Seed and test sets**: a fixed number of includes and excludes is drawn at
  random as the seed (retrieval examples); everything else is the test set.
  Xu: 10 + 50 seed papers; the test set keeps all remaining includes plus 200
  random excludes. Donners: 5 + 30 seed papers; the test set keeps everything.
- **Model**: Azure OpenAI deployment `gpt-5-mini`, temperature 0. Embeddings:
  `text-embedding-3-small`, stored in Chroma.
- **Leakage guards**: test papers never enter Chroma, and every RAG run checks
  that the stored examples come from the same split and share no paper with
  the test set.
- **Metrics**: *Recall (strict)* counts unsure as not included. *Recall
  (unsure kept)* treats unsure as going to a person, so only excludes count as
  misses. *Final lost* counts papers in the final review that the model
  excluded. *(adj)* corrects for downsampled excludes.

## Limits

- The retrieval finding rests on one review (Xu_2022). Donners agrees in
  direction but has too few includes (9) and runs to confirm it.
- The two Xu splits overlap: each test set keeps every non-seed include, so
  about 66 of 76 includes are shared. The splits mainly differ in which
  examples retrieval can see.
- The agent was run on split 2 only, twice.
- Results cover the open-access subset with abstracts, not every record the
  reviewers screened.
- Xu is scored against title/abstract decisions and Donners against final
  decisions, so their numbers are reported separately and not pooled.
- Decisions vary between identical runs even at temperature 0 (on Xu, up to 12
  unsure papers between runs), so single runs are not comparable.
- Xu_2022 belongs to SYNERGY's "train" group and was used to choose settings.
  Confirmation on a "test" group review, with nothing changed afterwards, is
  still to do.

## Next steps

1. Confirm on a SYNERGY "test" group review with enough includes, using the
   settings fixed here.
2. Repeat the Donners RAG and baseline runs so it has three tagged runs per
   setup.
3. Try the number of retrieved examples (1, 3, 5) on Xu, and a seed-size
   experiment with a fixed test set.
4. Run the agent without its flag tool, to separate the effect of searching on
   its own from the effect of flagging.

## Reproduce

```bash
synergy get -d Xu_2022 -o data/synergy_xu
python -m src.load_synergy --dataset Xu_2022 --data-dir data/synergy_xu \
  --out-dir data/synergy/prepared/Xu_2022_rs7 --label abstract \
  --n-seed-include 10 --n-seed-exclude 50 --n-test-exclude 200 --random-state 7
python -m src.ingest --dataset Xu_2022_rs7
python -m src.evaluate --dataset Xu_2022_rs7 --run-tag run1
python -m src.evaluate --dataset Xu_2022_rs7 --k-examples 0 --run-tag run1
python -m src.evaluate --dataset Xu_2022_rs7 --agent --run-tag run1
python -m src.summarize_runs --dataset Xu_2022_rs7
```

Repeat the evaluate commands with `run2` and `run3`. `synergy get` only writes
into an empty folder, so use a separate output folder per download.
