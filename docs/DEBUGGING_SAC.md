# SAC Training Debugging Guide

## Critical Issues Found:

### 1. **Replay Buffer Too Small (CRITICAL)**
- **Current:** `capacity=500` 
- **Problem:** With 2048 envs training every step, you're discarding data after ~0.25 steps per env!
- **Fix:** Increase to at least 100,000 (ideally 500k-1M for long training)
```python
buffer = BatchedReplayBuffer(NUM_ENVS, capacity=500_000, obs_dim=obs_dim, act_dim=act_dim)
```

### 2. **Train Per-Step = 0.0000 (CRITICAL)**
- **Problem:** Rewards showing as zero - either logging bug or environment not rewarding
- **Debug:**
  - Check if `next_env_state.reward` is actually non-zero
  - Verify environment gives rewards (might be sparse)
  - Check if reward scaling is applied before logging
- **Fix:** Log raw rewards BEFORE scaling, and also log scaled rewards separately

### 3. **Reward Scaling Too Aggressive**
- **Current:** `batch.rew * 0.1` (for HumanoidRun)
- **Problem:** Go1JoystickFlatTerrain might not need this, making rewards too small
- **Fix:** Try removing scaling or using 1.0 for Go1:
```python
# For Go1JoystickFlatTerrain, try:
batch_scaled = batch.replace(rew=batch.rew * 1.0)  # No scaling, or try 0.5
```

### 4. **Target Network Update Too Slow**
- **Current:** `polyak=0.995` (0.5% update per step)
- **Problem:** Very slow target updates, especially early in training
- **Fix:** Consider using faster updates early:
  - Try `polyak=0.99` for faster learning
  - Or use `polyak=0.005` (update rate, not retention) for hard updates less frequently

### 5. **Warmup Too Short**
- **Current:** `WARMUP_STEPS = 100`
- **Problem:** With 2048 envs, only ~0.05 steps per env before training
- **Fix:** Increase to at least 1000-5000 steps:
```python
WARMUP_STEPS = 5000  # ~2.4 steps per env minimum
```

### 6. **Q-Values Very Negative**
- **Problem:** Q-values around -0.12 to -0.01 suggest:
  - Critic initialization might be off
  - Target network not updating fast enough
  - Rewards too small (scaling issue)
  - Network architecture issues

### 7. **No Episode Completions**
- **Problem:** 0/10 eps completed suggests episodes are very long or never terminate
- **Debug:** 
  - Check episode length limits
  - Verify `done` signal is working
  - Consider if environment has sparse rewards/terminations

## Recommended Fixes (Priority Order):

### Immediate (High Impact):
1. **Increase replay buffer capacity** to 500k
2. **Remove or reduce reward scaling** for Go1 (try 1.0 or 0.5)
3. **Increase warmup** to 5000 steps
4. **Fix reward logging** to show raw rewards

### Medium Priority:
5. **Faster target updates** (`polyak=0.99` or `polyak=0.005`)
6. **Tune learning rate** - try 5e-4 or 1e-3
7. **Check network initialization** - verify weights are reasonable

### Debugging Steps:
8. **Add reward logging:**
```python
# Before scaling
metrics["per_step_reward_raw"] = next_env_state.reward.mean()
metrics["per_step_reward_scaled"] = (next_env_state.reward * 0.1).mean()
# After scaling
metrics["per_step_reward"] = batch_scaled.rew.mean()
```

9. **Log buffer statistics:**
```python
avg_buffer_count = jnp.mean(buffer_state.count)
metrics["buffer_avg_count"] = avg_buffer_count
metrics["buffer_min_count"] = jnp.min(buffer_state.count)
```

10. **Monitor Q-value range:**
   - Check if Q-values are in reasonable range (should match reward scale)
   - If Q-values are way too negative/positive, adjust initialization

11. **Compare with PPO:**
   - Check PPO hyperparameters (learning rate, network size, etc.)
   - Match any environment-specific settings

## Hyperparameter Comparison (SAC vs PPO):

| Parameter | SAC (Current) | PPO (Reference) | Recommendation |
|-----------|---------------|-----------------|----------------|
| Replay Buffer | 500 (❌ TOO SMALL) | N/A | 500k-1M |
| Learning Rate | 3e-4 | 3e-4 | ✓ OK |
| Batch Size | 256 | 1024-2048 | Consider increasing |
| Network Size | (256, 256) | Varies | ✓ OK |
| Warmup | 100 (❌ TOO SMALL) | N/A | 5000+ |
| Reward Scaling | 0.1 | Varies | Try 1.0 for Go1 |

## Quick Wins:
1. Change `capacity=500` → `capacity=500_000`
2. Change `batch.rew * 0.1` → `batch.rew * 1.0` (for Go1)
3. Change `WARMUP_STEPS = 100` → `WARMUP_STEPS = 5000`
4. Add reward logging to debug the 0.0000 issue

