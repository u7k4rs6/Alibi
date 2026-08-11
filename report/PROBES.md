# Hyperparameter probes

Ten steps each on `Qwen/Qwen3-0.6B` with the chat template, arm a0 only. **Probes, never evidence.** Excluded from the evidence index by run-id prefix and by policy.

| Probe | lr | beta | Reward step 0 to last 3 | Holds or rises | Entropy first3 to last3 | Capped | KL mean | Status |
|---|---|---|---|---|---|---|---|---|
| A-current | 1e-05 | 0.0 | run produced no steps | n/a | n/a | n/a | n/a | subprocess_exit |
| B-lr-10x-lower | 1e-06 | 0.0 | run produced no steps | n/a | n/a | n/a | n/a | subprocess_exit |
| C-kl-anchor | 1e-05 | 0.02 | run produced no steps | n/a | n/a | n/a | n/a | dirty_git_tree |

**Chosen: none.** no condition held reward flat or rising, so the stage was not launched

## Did a positive advantage raise the logprob?

This could not be recovered from v1's stored data. `trainer_logprob` was written as a literal copy of `sampler_logprob` rather than a recomputation under the updated policy, and no prompt repeats within a run, so there was no same-prompt comparison to make. Measured directly in the probes instead: each step's completions are re-scored under the policy **after** that step's update, and the sign of the logprob change is compared against the sign of the advantage.

| Probe | Completions compared | Sign agreement |
|---|---|---|
| A-current | not measured | no advantage checks were recorded |
| B-lr-10x-lower | not measured | no advantage checks were recorded |
| C-kl-anchor | not measured | no advantage checks were recorded |

Agreement near 0.5 would mean the update is not moving logprobs in the direction the advantage asks for, which would point at the optimiser or the loss rather than at the reward. Agreement well above 0.5 means the gradient is being applied as intended and the problem, if any, is upstream in what the reward is rewarding.
