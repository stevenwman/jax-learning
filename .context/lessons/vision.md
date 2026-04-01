# Vision RL Lessons

## Frame Stacking: Locomotion ≠ DMC

The DMC/manipulation pixel-RL community (DrQ-v2, CURL) stacks 3 raw RGB frames along channels → one CNN pass. This is the standard for off-policy methods on simple control tasks.

The locomotion community (ANYmal, DeFM, LocoMamba) does NOT use raw frame stacking. They use CNN per-frame → concat with proprioception → GRU/RNN. Reason: locomotion already has rich low-dim proprioceptive signal; vision provides terrain/obstacle geometry, not motion context. Temporal memory over 10+ steps matters for terrain estimation, which raw 3-frame stacking can't provide.

**DreamWaQ and Walk These Ways are not pixel methods.** They use proprioception + learned terrain estimators (latent context). No camera, no frame stacking.

## Asymmetric Critic Simplifies Vision Architecture

With a privileged critic (critic sees 122d state, actor sees pixels), the standard DrQ-v2 shared-encoder pattern doesn't apply. The critic never touches images, so:
- No frame stacking on critic side
- No shared encoder with stop-gradient on actor (the DrQ-v2 trick)
- Actor's CNN trains purely from policy gradients (or with auxiliary loss)
- Simpler implementation, cleaner gradient flow

This is consistent with Pinto et al. 2018 (Asymmetric Actor Critic for Image-Based Robot Learning).

## Pixel Replay Buffer: uint8 is Non-Negotiable

DrQ-v2 stores individual frames as uint8 and assembles stacks at sample time. At 84×84×3, each frame is ~21KB as uint8 vs ~85KB as float32. A 100K-entry buffer at 3-frame stacks: ~6.3GB (uint8) vs ~25GB (float32). On 16GB GPU, float32 is impossible.

## MJWarp Renderer: Warp-Only, Fixed nworld

`mjx.render()` only works with `impl="warp"`. Pure-JAX MJX cannot render. The `nworld` parameter is fixed at `create_render_context()` time — must match training batch size. This means you can't dynamically change env count after renderer init.
