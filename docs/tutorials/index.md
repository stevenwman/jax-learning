# Tutorials

Step-by-step guides for common tasks. Tutorials assume you have completed [Installation](../getting-started/installation.md) and the [Quickstart](../getting-started/quickstart.md).

- [**Train Locomotion**](train-locomotion.md) — Train a Go2 joystick policy with FastSAC on GPU-parallel Warp envs
- [**Train PushT**](../api/pusht.md#training-from-scratch-working-recipe) — Planar-pushing manipulation: working RL-from-scratch recipe (keypoint obs, frame stacking, contact-gated reward, TimeLimit). Reference on the PushT API page.
- [**Custom Environment**](custom-env.md) — Add a new MuJoCo environment using the bongo board handstand task as a worked example
- [**Custom Rewards**](custom-rewards.md) — Define reward and observation terms using the RewardSpec / ObsSpec systems
- [**Sim-to-Real**](sim2real.md) — Export a trained checkpoint to ONNX and deploy it on a real Unitree Go2
- [**Asymmetric Critic**](asymmetric-critic.md) — Give the critic privileged observations at training time while keeping the actor deployment-ready
