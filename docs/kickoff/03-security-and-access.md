# Alibi: Security and Access

Companion to `01-prd.md` and `02-technical-architecture.md`.

Two things make this project's security surface non-trivial despite it being a small research repo: it executes untrusted generated code thousands of times, and it publishes artifacts that are meant to be checked by strangers.

---

## 1. Threat model

**In scope:**
- Model-generated Python doing something destructive, expensive, or network-reaching when executed for scoring.
- Credentials leaking into the repo, into artifacts, or into logs that get published.
- Published artifacts containing absolute paths, machine names, or anything else personal.
- A reader being unable to tell whether a published artifact came from a clean tree.

**Out of scope:**
- A 0.5B LoRA adapter being dangerous. It is not. It is a model that learned to hardcode unit test inputs.
- Adversarial attack on the training infrastructure by a third party. This is a single-operator research repo.

## 2. Executing generated code

The generated code is untrusted input, and it is untrusted input written by a system being actively rewarded for finding unintended shortcuts. Treat it accordingly.

**Required controls:**

| Control | Implementation |
|---------|----------------|
| Process isolation | `subprocess` with a dedicated runner script. Never `exec` or `eval` inside the trainer process. |
| Wall clock limit | Hard timeout, 5 seconds default, kill the process group not just the child. |
| CPU limit | `resource.RLIMIT_CPU` |
| Memory limit | `resource.RLIMIT_AS`, 512 MB default |
| File size limit | `resource.RLIMIT_FSIZE`, small |
| Process limit | `resource.RLIMIT_NPROC`, blocks fork bombs |
| Filesystem | fresh `tempfile.mkdtemp()` per execution, `cwd` set to it, removed after |
| Network | no network. On Linux, run under a network namespace with no interfaces, or at minimum block at the sandbox layer and assert the block in a test |
| Imports | denylist enforced in the runner preamble: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib` write paths, `ctypes`, `importlib`. Prefer a small allowlist if it does not break MBPP solutions. |
| Environment | scrubbed env, no inherited API keys, no HOME |

**Fail-closed rule:** if the sandbox cannot be established, the run aborts. It never silently degrades to an unsandboxed execution, and there is no flag to make it do so.

**Test it.** `tests/test_executor.py` includes an actual fork bomb, an actual infinite loop, an actual `open('/etc/passwd')`, and an actual socket connect. A sandbox with no adversarial test is a comment, not a control.

**Note for Kaggle and Colab:** these run as root in a container with network access. Namespace isolation may not be available. If it is not, document precisely which controls are active on that host in `env.lock`, and do not claim isolation the environment does not provide.

## 3. Credentials

**Secrets in use:** a hosted inference API key for the monitor, optionally a Hugging Face token, optionally a W&B key.

**Rules:**
- Read from environment variables only. Never a config file, never a notebook cell, never a default argument.
- `.env` is gitignored on day 1, before the first commit, along with `artifacts/**/raw_responses/` if raw monitor responses are ever stored unscrubbed.
- Install `gitleaks` or `detect-secrets` as a pre-commit hook on day 1. This is ten minutes and it is the single control that prevents the worst realistic outcome.
- If a key is ever committed, rotate it. Do not rewrite history and consider it handled.
- Kaggle and Colab secrets go through their secret managers, not pasted into cells that get saved into version history.

## 4. Publishing artifacts safely

Everything in `artifacts/` is intended to be public and read by a mentor. Scrub before it lands.

**Scrubbed at write time, not at publish time:**
- Absolute paths become repo-relative. Lockstep already does this in its subject-environment capture; reuse the approach.
- Usernames, home directories, hostnames.
- API keys, obviously, including any that appear inside a monitor's echoed error message.

**Kept deliberately:**
- Git revision and dirty flag. A dirty-tree artifact stays labelled dirty rather than being quietly cleaned. An artifact produced from an uncommitted working tree is a known failure mode.
- Full monitor prompts and templates. The monitor prompt is a load-bearing methodological choice and hiding it would make the result uncheckable.
- Failed and null runs.

**Provenance honesty:** state in `artifacts/index.json` what provenance actually rests on. If mtimes are not tamper-evident and no tree hash was captured, say so rather than implying stronger guarantees. The defensible claim is usually narrower than the tempting one.

## 5. Licensing and attribution

Verify each before publishing rather than trusting recollection:

- **MBPP:** check the dataset card for the current license and attribution requirement.
- **EvalPlus / MBPP+:** check the repo license and cite the paper.
- **Qwen2.5-0.5B / 1.5B:** check the model card license terms for redistribution of derived adapters.
- **TRL, PEFT, transformers:** Apache 2.0, note in the README.
- **Monitor model:** check the hosted provider's terms on publishing model outputs. Monitor verdicts are being committed to a public cache, which is a redistribution of model output.

Put a `LICENSES.md` in the repo naming each, with the version or revision checked and the date checked. This takes fifteen minutes and it is the kind of thing a DeepMind mentor notices the absence of.

## 6. What gets released

**Released:** code, LoRA adapters, all artifacts, the monitor verdict cache, the report, the figure, the pre-registration tag.

**Not released:** API keys, raw provider responses if they contain account identifiers, anything under `.env`.

**Dual-use note, one paragraph in the README:** this trains a small model to evade a text-based monitor on a toy coding task. The artifact has no capability worth misusing, and the technique it demonstrates is already published. It is included for the same reason the safety literature publishes it, which is that the mitigation depends on knowing the failure mode exists. Keep this short and unsanctimonious. One paragraph, not a section.

## 7. Access

Single operator, no collaborators this week. Therefore:

- Public GitHub repo from the first commit. Nothing here needs to be private and a private repo cannot be linked in an application.
- Branch protection is unnecessary at one operator. Signed commits are optional.
- Compute accounts (Kaggle, Colab, any rented host) are personal. If a host is rented, terminate it explicitly and confirm termination, since a forgotten instance is the only real financial risk in the whole project.

## 8. Day-one checklist

Do all of these before the first training run, not after.

- [ ] `.gitignore` covering `.env`, `__pycache__`, `*.pt`, `wandb/`, local scratch
- [ ] Secret-scanning pre-commit hook installed and verified by attempting to commit a fake key
- [ ] Sandbox runner written, with the four adversarial tests passing
- [ ] Fail-closed assertion: sandbox unavailable aborts the run
- [ ] Path scrubbing in the logger, verified on one real artifact
- [ ] `LICENSES.md` filled in with dates checked
- [ ] CI workflow file exists and runs the test suite on CPU
