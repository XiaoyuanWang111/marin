# Campaign analysis

`snowball_tournament/pairwise.py` selects one Snowball-lineage checkpoint from
all Snowball and Grug candidate rows in the campaign tracker. It requires a
complete observed candidate panel after excluding FinanceBench, PIQA, and
TriviaQA. A benchmark is decisive only when the adjusted 95% confidence
interval for the score difference excludes zero; all other outcomes are ties.
When the durable evidence of both cells names the items behind each score
(corrected replay trials, or archive samples whose metric reproduces the
tracker score) and both cover the same items, the interval is the paired t
interval of the mean per-item difference; repeated replies to one item enter
through their mean. Other comparisons, such as those with a recovered
aggregate, use `1.96 x hypot(adjusted SEMs)`, and the outcome's
`difference_basis` says `unpaired`. Both intervals include the policy's
`1 / sqrt(0.9)` infrastructure-failure tolerance. Comparisons use the
full-precision aggregate recovered from durable trial evidence; the tracker's
three-decimal score is presentation-only. The tool writes the benchmark-level
decisions, complete pairwise table, ranking, cell statistics, and input hashes.

`critical_difference/plot.py` adapts the September 17 campaign's
`release-figures/generate_release_figures.py`. It plots the selected checkpoint
against the 13 non-Snowball baselines in `critical_difference/baseline_flops.csv`.
It reads each scored archive and any audited recovered metrics, then writes the
cell-level uncertainty table, FLOP table, rank tables, and uncontrolled and
FLOP-controlled critical-difference figures. The output README records the
common benchmark panel and tracker-snapshot hash.

`sotopia_length_control/rank.py` reads every sealed SOTOPIA-hard cell and its
episode summaries, computes normalized seven-dimension overall scores and the
evaluated agent's emitted character count, then fits the campaign's
AlpacaEval-style cohort GLM. It writes raw and length-controlled rankings plus
the exact tracker hash, source archives, selected regularization strength,
cross-validation losses, per-episode adjusted scores, and task-level SEM. Pass
the immutable object-store destination with `--output-prefix`; the generated
`tracker_updates.csv` maps each original archive to its durable regrade record.
Goal achievement remains available in the source metrics but is not the default
ranking outcome.

Run the tournament before plotting. Both commands and their required inputs are
specified in the campaign release document's “Reproduce the Snowball tournament
and critical-difference figures” section. These scripts report the results in
the supplied tracker; they do not attest policy conformance.
