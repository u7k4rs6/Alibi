# DECISIONS

Every autonomous decision, with what was chosen, what was rejected, why, and the
commit. Timestamps are UTC.

---

## 2026-08-10 preparation session

### D-01 Held-out timeout scaled per test, not per problem
**Chose:** a per-test SIGALRM budget inside the sandbox, plus a wall clock that
scales with the test count, plus RLIMIT_CPU scaled above the wall clock.
**Rejected:** a single larger flat timeout, which lets one pathological input
starve the other 104 and turn them all indeterminate.
**Why:** tasks 271, 392 and 599 timed out at a flat 5 s and would have published
as correct code that does not generalise. All three now pass with zero
indeterminate. Reference indeterminate rate across the full set is
121/39251 = 0.0031.
**Commit:** 5d091e9

### D-02 A candidate that does not compile is FAIL, not INDETERMINATE
**Chose:** compile and execute the candidate separately from the harness. A
candidate that fails to compile or raises while defining itself fails every
test.
**Rejected:** treating any missing outcome as indeterminate.
**Why:** the first smoke run tripped the 5 percent indeterminate halt at step 0,
because a 0.5B model emits malformed code routinely and that was being counted
as "the sandbox could not tell". A syntax error is a definitive statement about
the candidate. Only the sandbox failing to answer is indeterminate.
**Commit:** see `fix(env)` for the runner split.

### D-03 Interpreter named by real path inside the sandbox root
**Chose:** `os.path.realpath(sys.executable)` in the sandbox command, and the
venv prefix added to the bind list.
**Rejected:** leaving the mount namespace degraded under the venv.
**Why:** a venv's `bin/python` is a symlink chain that does not resolve inside
the minimal root, so `execv` failed and the Executor self test correctly
degraded to language-layer confinement. With the fix the mount namespace is
active during training, which is the strongest filesystem control available.

### D-04 KL guarded against non-finite values
**Chose:** skip completions shorter than two tokens and drop non-finite KL terms.
**Rejected:** clamping, which would invent a number.
**Why:** the first smoke run reported `kl inf`, which would have made the KL
spike halt condition either never fire or always fire.

### D-05 Reward isolation test narrowed to the real invariant
**Chose:** reward.py may not import or name the oracle; ScoredCompletion may not
carry it; no call site may pass oracle data into `reward_fn`; and in loop.py the
reward must be computed before the oracle is consulted.
**Rejected:** scanning every module under `alibi/train/` for oracle mentions,
which the test originally did.
**Why:** architecture doc section 3.3 explicitly requires the oracle verdict to
be attached downstream in the logging record, and `loop.py` is that downstream.
The broad scan forbade the logging the document requires. The narrowed version
is more precise and still fails if the oracle reaches the reward.

### D-06 Structural check extended, and demoted to diagnostic
**Chose:** extend to dict-keyed, sequence-lookup and membership forms, and
register the behavioural oracle as primary with structural as diagnostic only.
**Rejected:** leaving the PRD rule as written.
**Why:** day 1 measured the rule catching the if-chain cheat 372/376 and the
behaviourally identical dict cheat 0/376. That it needed extending at all is
evidence it can be evaded by a form nobody enumerated, so it must not be the
primary oracle.
**Commit:** 5d091e9

### D-07 Held-out scoring decoupled from the training critical path
**Chose:** a thread pool keyed by completion id, joined before the step record
is written. Sampling rate 1.0.
**Rejected:** subsampling during training.
**Why:** held-out never enters the reward, so it does not belong in the critical
path. Full coverage was kept because measured held-out scoring is fast enough at
this scale; the sampling rate is recorded per run so it can be reduced later
without ambiguity.

### D-08 Secret scanner baseline updated rather than the hook weakened
**Chose:** record the sha256 hashes in the split and eligibility manifests as
reviewed findings in `.secrets.baseline`.
**Rejected:** excluding those files from scanning, or disabling the plugin.
**Why:** they are hashes, not secrets, and a baseline is the tool's mechanism
for reviewed findings. Excluding the files would stop scanning them for real
secrets later.
