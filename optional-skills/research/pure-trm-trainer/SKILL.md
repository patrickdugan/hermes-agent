---
name: pure-trm-trainer
description: Build and run pure TRM controller training workflows under Hermes, including corpus assembly from TRM play logs, event logs, reasoning traces, normalized JSONL datasets, and hill-climbing search loops over generalization level. Use when Codex needs to curate cross-environment controller data, turn logs into conductor-style training corpora, launch a Hermes-wrapped trainer, or optimize a TRM bench by iterating on generalization breadth without mixing in prose or adapter authoring.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, TRM, Training, Generalization, Benchmarks]
    related_skills: [qmd]
---

# Pure TRM Trainer

Use this skill to keep TRM training narrow: controller behavior, transition choice, constraint satisfaction, verifier logic, and rebalance reasoning, not prose generation.

Start from normalized `state/tools/action` records whenever possible. If the source data is raw, convert it once into that shape before training.

## Workflow

1. Decide the target controller behavior.
   - Use TRM for topology, action selection, constraint satisfaction, rebalancing, and verifier-style reasoning.
   - Do not mix in story prose quality unless the user explicitly wants a hybrid model.
2. Arrange the corpus.
   - Read [data-arrangement.md](./references/data-arrangement.md) for source selection and split rules.
   - Read [source-patterns.md](./references/source-patterns.md) for concrete source recipes.
3. Build a normalized corpus.
   - Canonical builder:
     - `python hermes-skills/storyworld-conveyor/scripts/build_trm_training_corpus.py --config hermes-skills/storyworld-conveyor/sample_data/trm_training_corpus_spec.sample.json`
4. Launch Hermes-wrapped training.
   - Canonical runner:
     - `python hermes-skills/storyworld-conveyor/scripts/run_trm_trainer_hermes.py --config hermes-skills/storyworld-conveyor/sample_data/trm_trainer_hermes_safe.json`
   - Router bench runner:
     - `python hermes-skills/pure-trm-trainer/scripts/run_trm_routerbench.py`
     - Add `--dry-run` to resolve the portable bench config without launching.
     - Add `--template-root <path>` or `--corpus-spec <path>` if your local checkout lives outside the usual `C:/projects` or `/mnt/c/projects` roots.
   - Hill-climb runner:
     - `python hermes-skills/pure-trm-trainer/scripts/run_trm_generalization_hillclimb.py --config <search-spec>.json`
   - Smoke wrapper:
     - `powershell -ExecutionPolicy Bypass -File hermes-skills/pure-trm-trainer/scripts/run_trm_hillclimb_smoke.ps1`
     - Add `-RunId <name>` to stamp a run directory.
     - Add `-NoEval` to skip the evaluator command and rely on scorecard fallback from the trainer summary.
5. Inspect artifacts.
   - Check `prepare_corpus/manifest.json`
   - Check `trainer_config.resolved.json`
   - Check `launch_trainer/manifest.json`
   - Check final `summary.json`
   - For hill-climb runs, also check `hillclimb_ledger.jsonl` and each candidate `result.json`

## Hill-Climbing Loop

Use this when the user wants the best TRM bench score, not just a one-off trainer run.

1. Define the search axis as a generalization ladder.
   - Step 0: in-domain only, minimal state, no held-out envs.
   - Step 1: light generalization, a few held-out episodes.
   - Step 2: cross-run generalization within the same env family.
   - Step 3: cross-environment generalization across related families.
   - Step 4: broad generalization across families and action namespaces.
   - Step 5: hardest setting, strongest compression and weakest memorization.
2. Hold the evaluation anchors fixed.
   - Keep a small anchor set for the benchmark.
   - Do not tune on the anchors.
   - Preserve the same score definition across iterations.
3. Run one candidate at a time.
   - Change only a few knobs per iteration:
     - source mix
     - held-out breadth
     - state compression
     - negative-example rate
     - recovery-example rate
     - action-namespace normalization
   - Prefer a local mutation around the current best configuration.
4. Score the run.
   - Record train score, anchor score, failure rate, and recovery rate.
   - Track generalization gap.
   - Prefer the highest anchor score with an acceptable gap, not the highest train score.
5. Keep or mutate.
   - If anchor score improves, keep the candidate as the new baseline.
   - If anchor score drops but a different knob looks promising, branch a sibling candidate.
   - If repeated mutations plateau, move one rung up or down the generalization ladder.
6. Stop when the curve saturates.
   - Stop after the best anchor score fails to improve for several nearby mutations.
   - Prefer a smaller, cleaner dataset over a larger but overfit one.

## Search Heuristics

- Start narrow, then widen.
- Use one controlled mutation per loop when possible.
- Compare candidates on the same anchors.
- Treat train gain without anchor gain as overfitting.
- Favor a model that survives harder held-out envs over one that only spikes on easy repeats.
- Keep the final artifact set small and reproducible: corpus spec, resolved trainer config, run manifest, and summary.
- If you have a separate evaluator, have it write a compact `scorecard.json` with `train_score`, `anchor_score`, `failure_rate`, and `recovery_rate`.

## Corpus Rules

- Prefer diverse environments over repeated near-duplicates from one environment.
- Keep train/validation splits separated by environment or run when possible, not just by row.
- Preserve source metadata in `meta` so later audits can trace bad behavior back to a world, run, or prompt family.
- Treat reasoning traces as supervision only when they encode decision-relevant state, candidate actions, or critique. Skip decorative chain-of-thought.
- Use Monte Carlo, verifier, or transition metrics to decide what the controller should improve; do not guess from surface prose.

## Hermes Runner

Use [run_trm_trainer_hermes.py](../storyworld-conveyor/scripts/run_trm_trainer_hermes.py) for the full Hermes path. It performs three stages:

- `prepare_corpus`
- `prepare_config`
- `launch_trainer`

Use [run_hrm_trainer_hermes.py](../storyworld-conveyor/scripts/run_hrm_trainer_hermes.py) only when the user already has a clean conductor dataset and does not need corpus assembly.

## Source Types

Supported directly by the corpus builder:

- `trm_play_root`
  - Expects AICOO/TRM setup output with `worlds/*/trm_traces.jsonl`
  - Converts play logs into `state/tools/action`
- `jsonl`
  - `mode=normalized`: rows already contain `state`, `tools`, `action`
  - `mode=reasoning_trace`: build `state` from selected fields and map action/tool fields explicitly

## Hard Rules

- Keep this skill pure TRM. If the user wants prose, authoring adapters, or encounter block generation, switch to another skill or a separate workflow.
- Do not claim a run succeeded unless the Hermes stage manifests say `completed`.
- Do not train on mixed-quality logs without preserving source metadata.
- Do not let validation rows come from the same environment slice used for training if the task is cross-environment generalization.
- Do not silently strip invalid rows; record counts in the corpus manifest.
- Do not optimize on anchor evals and training evals at the same time; anchors stay fixed while candidates mutate.
- Do not let the search loop drift into generic hyperparameter tuning if the user asked for generalization-level search.

## References

- Data curation guide: [data-arrangement.md](./references/data-arrangement.md)
- Source recipes: [source-patterns.md](./references/source-patterns.md)
- Hill-climb search spec: [hillclimb-spec.sample.json](./references/hillclimb-spec.sample.json)
- Hill-climb smoke spec: [hillclimb-spec.smoke.json](./references/hillclimb-spec.smoke.json)
- Router bench spec: [routerbench-spec.json](./references/routerbench-spec.json)
- Router bench launcher: [run_trm_routerbench.py](./scripts/run_trm_routerbench.py)
- Local scorecard evaluator: [evaluate_trm_scorecard.py](./scripts/evaluate_trm_scorecard.py)

## Example Requests

- "Use pure-trm-trainer to turn these TRM play logs into a cross-world controller dataset."
- "Build a Hermes-safe TRM training run from verifier traces and held-out validation worlds."
- "Prepare a hub-ready pure TRM workflow that learns from logs and reasoning traces without mixing in prose generation."
