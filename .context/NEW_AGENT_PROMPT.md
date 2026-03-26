# New Agent Onboarding Prompt

Copy-paste this when starting a fresh agent session.

---

Read .context/AGENT_HANDOFF.md thoroughly before responding. This is your onboarding document — it contains everything you need to know about the project, the user, the codebase, and how to work effectively.

Key context for current state:
- Go2 PPO Phase A is DONE (robot walks, eval 233, seed 2100)
- SAC Phase B is in progress — eval ~142 at 16M steps, climbing slowly
- Root cause of the entire Go2 debugging saga: reward rebalancing (tracking 10x), not physics or algo bugs. Full trail in .context/go2/ppo_debugging.md
- Codebase was significantly cleaned up: 5 root scripts, consolidated train_offpolicy.py, .context/go2/ subfolder
- All docs are current as of 2026-03-26

Important operational patterns:
- Always use `uv run python` (not python3)
- Go2 env returns dict obs: {"state": 48d, "privileged_state": 122d}
- train_ppo_fast.py for PPO (lax.scan, 80k sps), train_offpolicy.py for SAC/TD3
- record_video.py saves _traj.npz + command arrow overlay — use trajectory data for diagnosis, not video analysis
- CheckpointManager saves best policy to ckpt_dir/best/
- Nuclio (SAM ViT-H) may respawn on GPU — kill via `sudo docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h`

Read these files in order:
1. .context/AGENT_HANDOFF.md (full project context)
2. .context/TODO.md (what's next)
3. .context/go2/lessons.md (Go2-specific lessons)
4. .context/LESSONS.md (general framework lessons)

Say "Ready" and wait for instructions.
