# Training Stability Analysis

## Observed Issues

1. **Volatile Evaluation Returns**: Evaluation returns fluctuate wildly (e.g., 545.90 → 265.66 → 369.64 → 148.76 → 580.44 → 84.89 → 3.39)
2. **Policy Degradation**: Later epochs (50-52) show very low returns (0.98-3.39) despite Q-values stabilizing
3. **No Episode Completions**: Evaluation shows "0/10 eps completed", meaning episodes run for full 1000 steps without terminating
4. **Q-Value Convergence**: Q-values stabilize around 21.50, suggesting critic is learning but policy may be collapsing

## Root Causes

### 1. Reward Scaling Mismatch
- **Issue**: Training uses `batch.rew * 0.1` (10x reduction), but evaluation uses raw rewards
- **Impact**: Creates distribution shift between training and evaluation
- **Location**: `jax_rl/main.py:238`

### 2. Small Evaluation Sample Size
- **Issue**: Only 10 episodes evaluated per epoch
- **Impact**: High variance in evaluation estimates, making it hard to track true performance
- **Location**: `jax_rl/main.py:452`

### 3. No Learning Rate Schedule
- **Issue**: Constant learning rate (3e-4) throughout training
- **Impact**: May cause instability in later training when policy should fine-tune
- **Location**: `jax_rl/main.py:30`

### 4. Episodes Not Completing
- **Issue**: CartpoleSwingup episodes run for full 1000 steps without `done=True`
- **Impact**: Evaluation returns are cumulative rewards over 1000 steps, not true episode returns
- **Possible Causes**:
  - Environment max episode length is 1000
  - Policy is not reaching terminal states
  - Auto-reset might be interfering

### 5. Target Network Update Frequency
- **Current**: Polyak averaging with τ=0.995 every step
- **Status**: This is correct for SAC, but very slow updates (0.5% per step)
- **Impact**: Target network lags significantly, which is intentional but may contribute to instability

### 6. Potential Overfitting
- **Symptom**: Q-values converge but policy performance degrades
- **Possible Causes**:
  - Overfitting to replay buffer distribution
  - Distribution shift between training and evaluation
  - Actor collapsing to suboptimal policy

## Recommendations

### Immediate Fixes

1. **Remove or Match Reward Scaling**
   ```python
   # Option A: Remove scaling for CartpoleSwingup
   if ENV_NAME != 'Go1JoystickFlatTerrain':
       batch_scaled = batch  # No scaling
   else:
       batch_scaled = batch.replace(rew=batch.rew * 0.1)
   
   # Option B: Apply same scaling in evaluation (not recommended)
   ```

2. **Increase Evaluation Sample Size**
   ```python
   # Change from 10 to 50-100 episodes
   avg_episode_return, num_episodes_completed, avg_episode_length = evaluate_policy(
       eval_actor, num_episodes=50, max_steps=1000, eval_seed_offset=epoch
   )
   ```

3. **Add Learning Rate Decay**
   ```python
   # Add to configuration
   LR_START = 3e-4
   LR_END = 1e-5
   LR_DECAY_STEPS = TOTAL_STEPS
   
   # Create schedule
   lr_schedule = optax.linear_schedule(
       init_value=LR_START,
       end_value=LR_END,
       transition_steps=LR_DECAY_STEPS
   )
   
   # Update optimizers
   actor_opt = nnx.Optimizer(actor, optax.adam(lr_schedule), wrt=nnx.Param)
   critic_opt = nnx.Optimizer(critic, optax.adam(lr_schedule), wrt=nnx.Param)
   ```

4. **Add Gradient Clipping**
   ```python
   # In train_step, after computing gradients
   a_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), a_grads)
   c_grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), c_grads)
   ```

5. **Monitor Additional Metrics**
   - Track policy entropy (already tracked as `entropy`)
   - Track action distribution statistics
   - Track Q-value spread (std dev)
   - Track gradient norms

### Medium-Term Improvements

1. **Adaptive Alpha (Entropy Temperature)**
   - Currently fixed at 0.2
   - Consider automatic entropy tuning to maintain target entropy

2. **Prioritized Experience Replay**
   - Current: Uniform sampling
   - Could help with sample efficiency and stability

3. **Delayed Policy Updates**
   - Update actor less frequently than critic (e.g., every 2 steps)
   - Can improve stability

4. **Target Network Update Frequency**
   - Consider updating target network less frequently (e.g., every 2 steps)
   - Or use higher polyak coefficient (0.999) for slower updates

### Debugging Steps

1. **Check Episode Completion**
   ```python
   # Add to evaluation
   print(f"Episodes completed: {num_episodes_completed}/{num_episodes}")
   print(f"Average episode length: {avg_episode_length}")
   # If all episodes run 1000 steps, check environment max_episode_length
   ```

2. **Visualize Policy Actions**
   - Plot action distributions over time
   - Check if actions are collapsing to boundaries

3. **Compare Training vs Evaluation Rewards**
   - Log raw training rewards (before scaling)
   - Compare with evaluation rewards
   - Should be similar if policy is consistent

4. **Check Replay Buffer Distribution**
   - Monitor buffer age (how old are samples being used?)
   - Check if buffer is dominated by old experiences

## Expected Behavior

After fixes:
- Evaluation returns should be more stable (less variance)
- Policy performance should improve or at least not degrade
- Q-values should continue to converge
- Episodes should complete (if environment allows)

## Next Steps

1. Remove reward scaling for CartpoleSwingup
2. Increase evaluation episodes to 50
3. Add learning rate decay
4. Monitor for 5-10 epochs
5. If still unstable, add gradient clipping
6. If still unstable, investigate action distribution and policy entropy

