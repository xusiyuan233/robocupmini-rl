# RobotcupMini project boundaries

- This repository is the user's RobotcupMini IsaacLab / PPO / AMP training project.
- Do not read, modify, run, stop, copy or publish projects or processes whose paths contain `lilin`.
- Other neighboring repositories are not automatically in scope. This snapshot explicitly excludes the separate Sim2Sim deployment project.
- Do not start training, change rewards, or interrupt GPU processes as part of repository maintenance without an explicit request.
- Never commit credentials, Conda directories, full logs, caches or checkpoints. Trusted model checkpoints are distributed separately as private Release assets.
- Keep approved expert data and audit bytes unchanged; hashes are checked by `style_reference_manifest.json`.
- When changing portable source intentionally, run `python -m tools.repro.update_manifest`, then the CPU checks documented in README. Do not use manifest regeneration to bypass failed demonstration validation.
