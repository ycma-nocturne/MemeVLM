# MemeBench reference coverage

Use a fresh checkout or extracted copy with the [CPU dependencies](../../README.md#set-up). Run from the repository root. Set `PREPARED` to the private frozen preparation folder and `RUNS` to its sealed judge run folder.

```sh
export PYTHONPATH="$PWD/code"
work=$(mktemp -d)
patch -d code -p1 < repro/chinese/numbered-span-judge.patch
python3 -m meme_study.reference_cli score --prepared "$PREPARED" --runs "$RUNS" \
  --allow-judge-json-fence --output "$work/frozen-scores"
python3 -m meme_study.reference_cli analyze --scores "$work/frozen-scores" \
  --seed 6000 --draws 10000 --output "$work/frozen-analysis"
python3 repro/chinese/sanitize_reference_coverage.py --scores "$work/frozen-scores" \
  --analysis "$work/frozen-analysis" --split test > "$work/coverage-test.json"

patch -d code -p1 < repro/chinese/all-citations-replay.patch
python3 -m meme_study.reference_cli score --prepared "$PREPARED" --runs "$RUNS" \
  --allow-judge-json-fence --allow-all-positive-spans --output "$work/replay-scores"
python3 -m meme_study.reference_cli analyze --scores "$work/replay-scores" \
  --seed 6000 --draws 10000 --output "$work/replay-analysis"
python3 repro/chinese/sanitize_reference_coverage.py --scores "$work/replay-scores" \
  --analysis "$work/replay-analysis" --split test > "$work/coverage-span-replay.json"
shasum -a 256 "$work/coverage-test.json" "$work/coverage-span-replay.json"
```

| Aggregate | SHA-256 |
| --- | --- |
| `coverage-test.json` | `12452aa11fdaa169db970694767d2272df3b230cc0481775352ff77bf65e76eb` |
| `coverage-span-replay.json` | `225c019b30ea00260262435e56fdc158985bc6e0c5ba2ec77b4afaa7a3a566ad` |

The post-hoc review uses private review packets and two sets of returned ratings. Set `REVIEW_HANDOFF` to the directory with the packets and `coordinator-only/` keys, and `RATINGS_DIR` to the directory with the returned CSVs.

```sh
python3 repro/chinese/summarize_human_audit.py --handoff "$REVIEW_HANDOFF" \
  --reference-records "$PREPARED/reference_records.jsonl" --ratings "$RATINGS_DIR" \
  --output "$work/human-audit-summary.json"
```

The default CSV naming scheme is `R1-explanations-all.csv`, with both reviewers and all three packet types. Use `--rating-suffix` if your six files have a different suffix after the reviewer and packet names. Use `--qa` with the separate QA CSV for the AI-assisted sensitivity analysis. Keep the returned forms and their hashes with the private results.
