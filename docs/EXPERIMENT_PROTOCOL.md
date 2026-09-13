# Full experiment protocol

## Freeze the data

Run the audit once, then create local filtered splits. Keep the generated
manifests with the paper artifact.

```bash
carl-audit-data --data-dir data --output records/data_audit.json
carl-filter-overlap --input-dir data --output-dir data_filtered
```

Set `data_dir` in the base configuration to `data_filtered`. Do not tune on the
test split. The filtered test scores form a new protocol and cannot be inserted
into the original-split table as if they were directly comparable.

## Run matched systems

Create the matrix and inspect every JSON before starting expensive jobs:

```bash
carl-make-matrix \
  --base-config configs/local.json \
  --output-dir configs/ablations
```

The three primary systems are `title_mle`, `context_mle`, and
`carl_learned_bleu`. `context_dropout_mle` is the simpler regularization
control. The attention-only and uniform policies test the learned selector;
the remaining runs vary one reward or policy-weight factor. Each
configuration runs seeds 13, 21, and 42 from the same base checkpoint.

## Evaluate context use

For every primary seed, run full, removed, and shuffled context conditions.
Use the same intervention seed for all systems.

```bash
carl-evaluate --config CONFIG --checkpoint CHECKPOINT --split test
carl-evaluate --config CONFIG --checkpoint CHECKPOINT --split test \
  --context-condition removed
carl-evaluate --config CONFIG --checkpoint CHECKPOINT --split test \
  --context-condition shuffled --intervention-seed 1729
```

## Aggregate and compare

Use `carl-summarize` for mean and sample standard deviation across seeds. Use
`carl-compare` only on prediction files with identical line IDs and references;
the command enforces both conditions before paired bootstrap resampling.

Archive, for every run: resolved configuration, runtime manifest, training
summary, best-checkpoint state, hypotheses, references, JSONL predictions,
metric files, and comparison files. Record any failed or restarted run rather
than silently replacing it.

## Paper update gate

Add a replication score only when all three primary seeds complete under the
frozen filtered protocol. Report mean and standard deviation, paired intervals
against context MLE, and the three context conditions. Keep these results
separate from the paper's original-split table. A single successful seed or a
tiny-model smoke test is software validation, not translation evidence.
