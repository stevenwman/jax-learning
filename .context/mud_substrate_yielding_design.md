# #10 — Substrate-yielding mud proxy (design note)

**Status:** design only, not built. The cheaper levers (4× DR + firm-plant + slow
reward) already SOLVE the Newton traverse, so this is about a MORE FAITHFUL training
substrate (better generalization / principled sim-to-Newton), not a blocker.

## Problem (from the force probe, 2026-06-11)
Newton bog = **traction/propulsion loss** on a yielding granular medium, NOT resistance.
Force probe: Newton bogs the robot at only **~3 N** total leg force (2% bodyweight),
while the analytic MudField needed **438 N** to bog it. The analytic field models the
WRONG physics: it adds an opposing wrench (resistance) on a SOLID floor that still gives
full push-off reaction. Newton's robot can't push off because the substrate flows.

The reward-shaping win (feet_slip penalty → firm planting) addresses traction
INDIRECTLY at the policy level. A substrate-yielding ENV proxy would address it at the
PHYSICS level → potentially better/cleaner transfer + a real "train on mud-like
dynamics" substrate.

## Goal
An MJX/Warp-traceable (vmap/jit-safe, mask-not-branch) proxy where a submerged foot's
PUSH-OFF YIELDS — tangential reaction is limited so leg drive doesn't fully convert to
body thrust — gated on foot depth, per-episode DR.

## Options (tangential = the load-bearing axis; Newton supports vertically ~ok)
1. **Low-friction DR.** Cheapest (friction DRSpec already exists; drop floor 0.3→0.1).
   REJECT as primary: mud is sticky-high-static-friction, not slippery — wrong mechanism
   (low friction = foot slides freely; mud = foot grips but the GROUND moves). Useful as
   a cheap robustness adjunct, not a faithful proxy.
2. **Yield-limited tangential contact (RECOMMENDED).** Cap the tangential force a
   submerged foot can exert ∝ (yield_stress · depth). Models granular shear yield
   (Herschel-Bulkley). Impl: in the substep, estimate the foot's commanded tangential
   ground force (from the controller torque via Jᵀ, already available in OSC path), and
   if it exceeds the mud yield cap, ADD a counter-force = excess (so net tangential ≤
   cap). Faithful: push harder → no extra thrust (yields). ~day build; the OSC controller
   already computes the foot Jacobian so the tangential projection is in reach.
3. **Tangential viscous drag** (opposes foot velocity). Already in MudField shear. This
   is RESISTANCE not traction-loss — keep as-is, not the fix.
4. **Sink + reduced normal support** (foot sinks ∝ depth, worse push-off geometry).
   Partial; doesn't capture tangential yield. Adjunct.
5. **Treadmill/flow proxy (NOVEL, closest to granular flow).** Under a submerged,
   pushing foot, give the CONTACT POINT a backward velocity ∝ push (substrate flows
   backward = foot spins in place like a sand treadmill). Impl: add foot velocity offset
   opposing the body's intended motion when the foot is loaded+submerged. Captures "push
   makes the ground move, not the body." ~day build; needs a load proxy.

## Recommendation
Build **Option 2 (yield-limited tangential)** first — it's the most physically defensible
(matches granular shear yield) and reuses the OSC foot-Jacobian already in
`go2_warp_components.OSC._run_osc`. Add `traction_loss=True` + `yield_stress` to the mud
config; implement the tangential cap in MudField.apply_substep (read controller tau,
project to foot tangential via Jᵀ, cap). DR the yield_stress per episode. Train the
slow+firm recipe on it, eval Newton — expect comparable-or-better transfer with a
physically honest substrate.

Validation gate: the trained policy's Newton final-y ≥ current slow+firm (y≈-0.36/-2.20)
AND the proxy's own force magnitude on the feet is O(Newton's ~3N tangential), not the
analytic 400N — i.e. confirm we matched the regime, not just added more force.

## Effort
~1 day (contact-force/Jᵀ projection is the fiddly part; the DR + variant + train + eval
scaffolding is all reused from the mud work). NOT a quick win — flagged for explicit go.
