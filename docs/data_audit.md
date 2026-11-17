# Pilot data audit (2026-11-14)

These are dataset composition checks, **not model results**.

| Extract | Rows | Coverage | Limitation |
| --- | ---: | --- | --- |
| BBQ pilot | 22 | 11 categories × 2 context conditions | All selected questions have negative polarity because this smoke test takes the earliest row in each stratum. No bias conclusion is valid from it. |
| BBQ balanced pilot | 44 | 11 categories × 2 context conditions × 2 question polarities | Still far too small for a stable category-specific estimate. |
| BOLD pilot | 40 | 8 each across gender, political ideology, profession, race, religious ideology | Balanced by top-level domain only; subgroup and category balance must be inspected before disparity estimates. |
| CrowS-Pairs pilot | 18 | 2 each across 9 bias types | Pair preference uses a simplified score, not the official metric. |
| StereoSet pilot | 16 | 4 each across gender, profession, race, religion | Intrasentence dev split only; composite StereoSet score is not yet implemented. |

The full BOLD source has 7,201 rows: gender 2,363, political ideology 498, profession 1,064, race 3,196, and religious ideology 80. An unweighted average over all rows would therefore be dominated by race and gender. Future group comparisons should report per-domain values and use a fixed weighting scheme.

The BBQ pilot answer labels are 8 A, 8 B, and 6 C; 11 contexts are ambiguous and 11 are disambiguated. All source dataset rows are kept out of Git; each JSONL file is hash identified in model predictions.

The synthetic membership corpus has 192 candidate records: 96 members and 96 nonmembers, balanced at 32 per side for each copy count (1, 4, 16). The training file has 1,184 rows, including 512 distinct routine records. Member and nonmember mean lengths are 213.06 and 212.85 characters. A five-fold character n-gram text-only classifier scored AUROC **0.4749** on this fixed split (seed 123). This is a useful initial confound check, not proof that every non-model feature is balanced or a privacy result.

The full BBQ extract contains 58,492 rows: 29,246 ambiguous and 29,246 disambiguated; 29,246 negative and 29,246 nonnegative questions. All ids are unique. Sixteen Gender Identity rows have `target_loc=NA` in the authors' metadata, so the bias scorer reports target-mapping coverage and excludes those rows from its signed stereotype score while retaining them for accuracy.

The full BOLD extract contains all 23,679 prompts listed by the official BOLD repository, rather than just the first prompt from each of 7,201 source records. The full CrowS-Pairs and StereoSet extracts contain 1,508 and 2,106 unique items respectively. Full pair preferences still use this project's documented AR log-likelihood versus diffusion masked-token scores; they are not the canonical benchmark scores.

Full prompt counts by BOLD domain are profession 10,195, race 7,657, gender 3,204, political ideology 1,984, and religious ideology 639. CrowS-Pairs ranges from 60 disability to 516 race-color items across nine types. StereoSet contains 962 race, 810 profession, 255 gender, and 79 religion items. Report per-group results and a clearly defined macro average alongside any micro average; otherwise the larger domains dominate.
