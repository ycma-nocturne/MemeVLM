# Rebuild the presentation

From `docs/final`, run with Node.js, `@oai/artifact-tool` from the Codex artifact runtime, and `jszip` available to Node.

```sh
work=$(mktemp -d)
node source/presentation/build.mjs evidence/full-results.json evidence/pilot \
  "$work/Final_Presentation.pptx" "$work/previews"
```

The build writes the editable PowerPoint, the speaker guide and PNG previews. Edit the main narration in `speaker-plan.json`. The human-audit appendix reads the published aggregates in `human-audit-public.json`. The pilot score input records two cue and relation outcomes for the illustrated Fitness Watch edit.

## Image provenance

The Cloud Backup, Fitness Watch and Role Reversal images in `assets/` are team-rendered controlled study examples. The opening memory contrast, the sarcasm and spiky-cactus examples and the natural-meme icon in the evaluation figure are original public schematics. They illustrate example logic and the evaluation design. Reported scores use the experimental inputs described in the report. Benchmark references appear in the slide notes.
