# Training Loop Analysis - Critical Issues Found

## Issues Identified

### 1. **CRITICAL: Actor Not Being Updated for Evaluation**
**Location**: Line 455
```python
eval_actor = actor  # actor is already the latest after scan_loop updates it
```

**Problem**: While `scan_loop` does call `nnx.update(models, final_state)` to update models in-place, the comment suggests this should work. However, we need to verify that the actor is actually being updated correctly.

**Fix**: The actor should be updated, but let's make it explicit by extracting from models tuple.

### 2. **Reward Scaling Mismatch**
**Location**: Line 246 (training) vs Line 325 (evaluation)

**Problem**: 
- Training uses: `batch_scaled = batch.replace(rew=batch.rew * REWARD_SCALE)` (0.1 scaling)
- Evaluation uses: Raw rewards from environment (no scaling)

**Impact**: This creates a distribution shift between training and evaluation. The policy is trained on scaled rewards but evaluated on raw rewards, which could explain the volatile evaluation returns.

**Fix**: Either:
- Remove reward scaling for CartpoleSwingup (set REWARD_SCALE = 1.0)
- Or apply same scaling in evaluation (not recommended)

### 3. **Buffer Sampling Before Buffer is Full**
**Location**: Line 236

**Problem**: The buffer is sampled immediately after adding transitions, even during warmup. This means:
- Early in training, the buffer contains mostly zeros or random warmup data
- The policy is being trained on low-quality data initially

**Impact**: This is actually normal for off-policy RL, but could contribute to instability early on.

### 4. **Evaluation Episodes Never Complete**
**Location**: Evaluation function

**Problem**: All evaluation episodes show "0/10 eps completed" and run for full 1000 steps. This means:
- Episodes are being truncated at max_steps
- The evaluation return is cumulative reward over 1000 steps, not true episode return
- This could explain volatility - different trajectories have different cumulative rewards

**Impact**: The evaluation metric is not a true episode return, making it hard to interpret.

### 5. **Potential Buffer Issue: Sampling from Partially Filled Buffer**
**Location**: shared_buffer.py line 134

**Problem**: When buffer is not full, we sample from `[0, count)`. This is correct, but:
- Early in training, `count` is small, so we're sampling from a very limited set
- This could lead to overfitting to early experiences

**Impact**: Could contribute to instability, especially early in training.

### 6. **No Validation of Buffer Contents**
**Problem**: There's no check to verify that:
- Transitions are actually being stored in the buffer
- Sampled transitions are non-zero (after warmup)
- Buffer count is increasing correctly

**Impact**: If the buffer isn't working, we'd never know.

## Most Likely Root Causes

1. **Reward Scaling Mismatch** - Training on scaled rewards (0.1x) but evaluating on raw rewards creates a fundamental mismatch
2. **Evaluation Metric Issue** - Episodes never complete, so we're measuring cumulative reward over 1000 steps, not true episode return
3. **Buffer May Not Be Working** - Need to verify transitions are actually being stored and retrieved

## Recommended Fixes

1. **Remove reward scaling for CartpoleSwingup** (set REWARD_SCALE = 1.0)
2. **Add buffer validation** to verify data is being stored
3. **Fix evaluation** to handle episode completion correctly
4. **Add debug logging** to track buffer state and sampled data quality

