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

## Answer Key (brief)
1. train_ppo_fast.py (PPO JIT), train_ppo.py (PPO fallback), train_offpolicy.py (SAC/TD3/Fast*), record_video.py, live_viewer.py
2. `uv run python train_offpolicy.py --algo sac --env Go2JoystickFlat --obs-norm`
3. Dict: {"state": 48d, "privileged_state": 122d}. Actor sees state, critic sees state too (off-policy). PPO critic sees privileged_state (asymmetric).
4. ckpt_dir/best/ — CheckpointManager compares eval_mean, saves when new high
5. None algorithmically. Fast uses lax.scan for collection (3-5x faster wall-clock). Same PPO.update().
6. PPO branch uses frozen_state + norm passthrough. SAC/TD3 branch extracts obs["state"] and adds batch dim.
7. Local-frame velocity command transformed to world frame. Rotates because the robot turns (yaw command changes heading).
8. Go2: .context/go2/lessons.md. General: .context/LESSONS.md
9. Crouching local optimum. Pose reward (~450) dominates tracking (~130). Fix: 10x tracking reward.
10. Problem is env/reward, not PPO implementation.
11. Policy can't commit to any strategy. Exploration prevents convergence. The reward landscape (not exploration) was the real issue.
12. Not a bug early — episodes are 1000 steps, scan is 20 steps, takes ~50 iters for first episode to complete. After that it should show values.
13. Eval function recreated as new closure each call → JIT recompiles. Fix: pass norm_state as arg via action_fn_kwargs.
14. Q1 is stable (good). Concern: entropy very negative, alpha tiny — SAC's entropy objective is barely active. Policy is near-deterministic, may limit further improvement.
15. Robot is sitting on its thighs/calves instead of standing on feet. Body geoms should be disabled (feetonly) or reward should incentivize standing.
16. Skill discovery (DIAYN/METRA) on real Go2 robot.
17. PPO is simpler, validates the env works. SAC is needed for DIAYN (wraps SAC).
18. No. PPO gets ~10 on HumanoidRun — wrong algo for high-dim. FastSAC gets 892.
19. PPO confirms the env produces walking (Phase A). SAC is Phase B.
20. Push back. The reward weights work (eval 233). Move to SAC Phase B or vision RL instead.
21. Find a working reference implementation and diff the FULL env code, not just config.
22. Push back. Go1 weights gave eval 12 on Go2. Reward balance differs per robot dynamics. Always check tracking/pose ratio > 0.5.
23. Push back. We tested 0.01/0.02/0.05 — higher entropy made it worse. The issue was reward balance, not exploration.
24. Push back. Over-engineering. The 4 algos have similar-enough interfaces. A closure in train_offpolicy.py handles the differences.
25. Push back. Two policies with identical eval=11.6 had opposite behaviors (standing vs crouching). Only trajectory data revealed this.
26. Push back. Diff the code first. We found vloss 0.25x + adv norm scope in 5 minutes of code reading.
27. Partial agree. Feetonly helps but the real fix was 10x tracking reward. Removing contacts alone didn't solve it.
28. Update journal, lessons (if reusable), TODO, debugging doc. In that order.
29. .context/go2/ppo_debugging.md — 19 hypotheses, run table, wrong hypothesis summary.
30. MJX physics NaN (not algo). Contact solver produces NaN/Inf at scale. Check env_setup.py NaN guard.
31. JAX_LOG_COMPILES=1 or time consecutive calls — first call ~100ms (compile), subsequent should be <1ms (cached).
32. Cheap sanity test first (kinematic sweep, manual joint command, CPU check). Never launch 50M runs to test a hypothesis you can verify in 30 seconds.
33. .claude/projects/.../memory/ as a memory file with frontmatter (type, name, description).
