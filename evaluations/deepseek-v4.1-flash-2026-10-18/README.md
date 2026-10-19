**English** | [中文](README.zh.md)

# Real API evaluation evidence

See [interpreted results](../../docs/llm-match-results.md) and [runner documentation](../../docs/llm-agent.md).

The report and logs come from three real `deepseek-flash`-supervised 420-second matches, seeds 9,000,000–9,000,002. All native replays verified 8,400 ticks and the recorded result. This is a one-red-W3 versus two-scripted-blue-robots scenario. Final results: one red win, two blue wins.

The `replay` basenames in `report.json` resolve in the original local output directory, `runs/deepseek-v4.1-flash-full/`, not in this evidence directory. Full native JSON is about 190 MiB per match and remains ignored/local. `sha256.json` records its SHA256 and location. The complete decision and per-second legal telemetry logs are included here; credentials and private reasoning transcripts are not.

The package version was bumped after evaluation; `execution-source-at-start.json` preserves the evaluated source fingerprint. `healthy_control` means integration/continuity passed, not tactical strength. Read errors, failed skills and availability separately. Arena-RL was private when this was recorded; it has been public since 2026-10-19.
