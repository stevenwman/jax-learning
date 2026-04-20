# New Agent Onboarding Prompt

---

Read .context/AGENT_HANDOFF.md for project overview. Then read .context/TODO.md for current priorities. Do NOT read all docs upfront — use the lookup pattern below.

## Doc lookup pattern (save context, read on demand)

The `.context/` folder is a graph of interconnected docs. Read only what you need:

```
AGENT_HANDOFF.md          ← START HERE (project overview, codebase map)
  └→ TODO.md              ← What's done, what's next
  └→ LESSONS.md           ← Index of framework lessons (click through, don't read all)
  └→ go2/                 ← Go2-specific docs (read only when working on Go2)
      ├→ lessons.md       ← Go2 gotchas (reward rebalancing, torque bug)
      ├→ ppo_debugging.md ← 19-hypothesis debugging trail
      ├→ sim_to_real_plan.md
      └→ sac_phase_b.md
  └→ lessons/warp.md      ← MuJoCo Warp lessons (joint ordering, PD gains, CCD)
  └→ references/vision_rl_design.md  ← Read only when working on vision
  └→ lessons/gpu_management.md       ← Nuclio docker fix, memory budget
```

**Rule:** Don't pre-load docs into context. When you encounter a topic (e.g., "why does Go2 use 10x tracking?"), grep `.context/` or read the specific file. Treat docs like a reference manual, not a textbook.

**Rule:** When you're about to make an assumption about past decisions, stop and check the docs first. The project memory file `feedback_check_docs.md` exists for this reason.

## Quick facts (always true)
- `uv run python` (not python3)
- No Co-Authored-By in commits
- Go2 dict obs: {"state": (48,), "privileged_state": (122,)}
- PPO: asymmetric (critic sees privileged_state). SAC/TD3: both see state. FastSAC uses asymmetric automatically when dict obs detected.
- Root train scripts: `train_ppo_fast.py`, `train_ppo.py`, `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`, `train_flashsac.py`, `train_pusht.py`. The 4 non-FlashSAC off-policy scripts are thin wrappers (~110–130 lines) that delegate to `jax_rl/training/offpolicy_loop.py::run_offpolicy_loop`. Also `record_video.py`. `archive/` holds legacy scripts (`train_offpolicy.py`, `live_viewer.py`, `record_video_cpu.py`).
- **Go2 envs:** `Go2WarpJoystickFlat` (Warp backend, unitree MJCF, Kp=20/Kd=0.5) is the primary benchmark env — eliminates sim2sim gap. Variants: `Go2WarpJoystickFlatTorqueSpeed`, `Go2WarpJoystickCurriculum`, `Go2WarpJoystickCurriculumTorqueSpeed`. Also `Go2BongoHandstand` (bongo board). Best reproducible eval: see `AGENT_HANDOFF.md` benchmark tables (post-truncation-fix 2026-04-13). **MJX Go2 env deleted 2026-04-09** — no longer in repo.
- **CRITICAL:** Warp env has joint→actuator ordering mismatch. `_act_to_joint` remap is essential. See `lessons/warp.md`.

Say "Ready" and wait for instructions.

---

# Verification Questions

Test the new agent's understanding before giving real tasks. Questions span technical recall, debugging reasoning, strategic alignment, and push-back ability. Some are intentionally misleading.

## Technical (codebase knowledge)
1. What are the 5 root-level Python scripts and what does each do?
2. How do you run SAC on Go2? Give the exact command.
3. What obs does the Go2 env return? What does the actor see vs the critic?
4. Where is the best checkpoint saved and how does it get there?
5. What's the difference between train_ppo.py and train_ppo_fast.py algorithmically?
6. How does record_video.py handle dict obs for SAC vs PPO checkpoints?
7. What does the command arrow in the video represent and why does it rotate?
8. Where are the Go2-specific lessons vs general framework lessons?

## Debugging (reasoning ability)
9. PPO on Go2 gets eval 12 and the trajectory shows calves saturated at -1.0 with base height 0.17m. What's happening and what's the fix?
10. Brax PPO and our PPO both get eval 11.6 on Go2. What does this prove?
11. You increased entropy_coef from 0.01 to 0.05 and eval got WORSE (2-4 vs 12-14). Why?
12. Training return shows 0.0 for the first 50 iterations of train_ppo_fast.py. Is this a bug?
13. Eval takes 40 seconds per call in train_ppo_fast.py. What's the likely cause and fix?
14. SAC on Go2 shows Q1=11.5, entropy=-5.5, alpha=0.004. Is this healthy? What's the concern?
15. You found 4/6 floor contacts at the crouching pose are from unnamed body geoms. What does this mean?

## Strategic (project alignment)
16. What's the north star for this project?
17. Why did we validate PPO on Go2 before trying SAC?
18. Should we add PPO support for HumanoidRun? Why or why not?
19. What's the prerequisite before training Go2 with SAC (not PPO)?
20. I want to spend a week perfecting Go2 reward weights. Thoughts?
21. What should you do FIRST when porting an env from one robot to another?

## Push-back (will they challenge bad ideas?)
22. Just copy Go1's reward weights to Go2 — the robots are similar enough.
23. The robot isn't walking, let's increase entropy_coef to 0.1 for more exploration.
24. Let's add a BaseAlgorithm ABC now that all algos share select_action.
25. Skip the trajectory analysis, just watch the video to see if it's walking.
26. Run 100M training steps to investigate a 2x sample efficiency gap instead of diffing the code.
27. Let's disable all body collisions in Go2 — Go1 uses feetonly so we should too.

## Operational (workflow habits)
28. You just finished a big debug session. What do you update before moving on?
29. Where would you find the full diagnostic trail for the Go2 debugging?
30. Training NaN'd at 50M steps. Q1 was healthy at 40.0 right before. What's your first hypothesis?
31. How do you verify a JIT recompilation bug?
32. You want to test if a physics change helps. Do you launch a 50M step run or something else first?
33. The user asks you to remember something for future sessions. Where do you save it?
