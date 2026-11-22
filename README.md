# LLM Judge Reliability Lab

An offline Python toolkit for auditing pairwise LLM-as-a-judge evaluations. It makes a common hidden failure measurable: a judge can appear accurate while preferring whichever answer is shown first.

## What it measures

For each evaluation item, record the judge's choice once in the original order and once with the candidates reversed. The report includes:

- Pairwise accuracy against a human/reference outcome, with a deterministic percentile-bootstrap 95% interval.
- Swap consistency after mapping the reversed presentation back to the original candidate identities.
- First-position selection rate across decisive votes. A rate far from 0.5 can indicate order sensitivity; inspect its interval and examples rather than treating it as proof of bias.
- Brier score and five-bin expected calibration error for the supplied confidence. Confidence is interpreted as the judge's estimated probability that its decision is correct; each paired presentation is scored separately.
- Reference outcome counts and confidence-bin support, so sparse results stay visible.

The tool validates JSONL strictly, rejects duplicate case IDs, and returns machine-readable JSON. It does not call a model, train a judge, invent human labels, or claim statistical significance from a small sample.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
judge-audit examples/judgments.jsonl --bootstrap 2000 --seed 17
```

Or run without installing the console script:

```bash
python -m judge_lab.cli examples/judgments.jsonl
```

The bundled example is a small hand-authored fixture demonstrating correct, incorrect, inconsistent, position-favoring, and tied outcomes. It is not a benchmark of any real model.

## Input format

One JSON object per line, with exactly these fields:

```json
{"case_id":"q-001","expected":"A","original":"A","swapped":"B","confidence":0.82}
```

- `expected`: reference winner (`A`, `B`, or `tie`) for the original candidate identities.
- `original`: selected displayed slot (`A`, `B`, or `tie`) when candidates are shown A then B.
- `swapped`: selected displayed slot after the candidates are shown B then A. The tool maps this vote back to the original identities before comparing it.
- `confidence`: a number in `[0, 1]`, interpreted as estimated probability that a decision is correct. The same confidence is assigned to each of the two presentations; supply one record per prompt/candidate pair.

Include a fresh, independent pair of judgments per case when estimating order effects. Replaying an identical cached response in both orders cannot reveal how a live judge behaves. Reference labels should be prepared independently of the judge under evaluation.

## Python API

```python
from judge_lab import audit, load_cases

cases = load_cases("examples/judgments.jsonl")
report = audit(cases, bootstrap_samples=2000, seed=17)
print(report["first_position_rate"], report["swap_consistency"])
```

`audit()` is deterministic for a fixed input, bootstrap count, and seed. Accuracy intervals resample paired cases (keeping the two presentations together); consistency and first-position intervals resample cases. The intervals do not correct for prompt clustering, annotator effects, multiple comparisons, or a nonrepresentative sample. Treat them as descriptive uncertainty summaries, not guarantees.

## Interpreting results

High accuracy alone is not enough: a judge may change its selected answer when the display order changes. Low swap consistency alongside a first-position rate above 0.5 suggests order sensitivity in this sample. A confidence score can also be poorly calibrated even when average accuracy looks acceptable. Review pair-level disagreements, use enough representative items, repeat with randomized order and prompt variants, and compare against independent human adjudication before relying on a judge for high-impact decisions.

## Scope and limitations

This is an offline analysis tool for already-collected pairwise judgments. It does not connect to an LLM provider, generate examples or labels, verify references, or establish that results transfer to other models, prompts, domains, languages, or production traffic. The bundled fixture is synthetic/illustrative, intentionally tiny, and unsuitable for drawing model-quality conclusions. The bootstrap interval assumes the sampled rows are the units of interest; adapt the resampling design for clustered or repeated prompts.

## Project quality

- Python standard library at runtime; no API keys or network access.
- Unit tests for vote normalization, ties, confidence metrics, deterministic intervals, validation, and empty input.
- GitHub Actions runs pytest and Ruff on Python 3.11, 3.12, and 3.13.

## License

MIT.
