# torch-sim autobatcher OOM fix — review overview

**Benchmark:** `examples/primary_amines_benchmark.py` (4272 primary-amine XYZs, `fire`, fconv 5e-2)
**torch-sim:** local editable checkout `dev-deps/torch-sim`, on latest `main` (`e4f4d4e`, fast-forwarded from upstream `TorchSim/torch-sim`).
**Scope of fix:** `torch_sim/` only. No ORB/UMA/gpuma changes.

---

## What fails, and what doesn't

| Model | Result (before fix) | Notes |
|---|---|---|
| uma-s-1p1 / uma-s-1p2 / uma-m-1p1 | pass | (an unrelated *external* process grabbed GPU1 mid-run and caused spurious UMA OOMs in the big sequential run — reproduced fine on a clean GPU) |
| orb-v3-direct | **pass** (4272/4272, mem-est ~26–33 s) | estimation never OOMs; caps at `max_atoms_to_try` |
| orb-v3-conservative | **crash** | `RuntimeError: Failed to allocate 2556 bytes on device 'cuda:N'` |

Only **orb-v3-conservative** crashes. It computes forces by autograd (≈2× memory), so the
autobatcher's memory-estimation probe drives the GPU to a genuine OOM, which the other models avoid.

## Root cause (two independent defects in `torch_sim/autobatching.py`)

The autobatcher estimates a safe batch size by replaying the model forward on
geometrically-growing batches until it OOMs (`determine_max_batch_size` →
`estimate_max_memory_scaler`, driven by `InFlightAutoBatcher`). Two bugs on that path:

1. **OOM signature too narrow + a control-flow bug.**
   - `determine_max_batch_size` only recognised the substring `"CUDA out of memory"`.
     ORB v3's neighbor lists run on Warp/`nvalchemiops`, whose allocator raises
     `"Failed to allocate <n> bytes on device '…'"` (`warp/_src/context.py:alloc`).
     That OOM was **not** recognised, so it propagated and killed the run *during estimation*.
   - Even passing a broadened *list* wouldn't have helped: the matcher's `raise` sat
     **inside** the `for msg in oom_error_message` loop, so the first non-matching entry
     re-raised before later entries were ever checked. The documented list feature was dead.

2. **PyTorch caching allocator starves Warp's separate pool after estimation.**
   After fixing (1), estimation survives — but the crash simply **moves** to the first *real*
   optimization forward pass (`fire_init`), failing on a tiny **2556-byte** Warp allocation.
   A 2556-byte failure is the fingerprint of allocator contention, not true exhaustion:
   the estimation forwards leave PyTorch's caching allocator holding ~all of device memory,
   and Warp/`cudaMallocAsync` allocates from a *separate* pool that then has nothing left.
   (orb-v3-direct dodges this because its estimation never reaches a hard OOM and uses less
   peak memory.)

Both must be fixed: (1) lets estimation survive the probe OOM; (2) frees the memory so the
real run — and Warp specifically — isn't starved.

## The fix (`torch_sim/autobatching.py`, ~20 functional lines)

- Add `DEFAULT_OOM_ERROR_MESSAGES = ("CUDA out of memory", "Failed to allocate")` and use it as
  the default `oom_error_message` for `determine_max_batch_size`, `InFlightAutoBatcher`,
  `BinningAutoBatcher`.
- Replace the `for … raise`-inside-loop with `if any(msg in exc_str …): return safe; raise`
  (re-raise happens only after *all* signatures are checked).
- At the end of `estimate_max_memory_scaler` (shared by both batchers), release memory:
  `torch.cuda.synchronize(); torch.cuda.empty_cache()` (guarded by `torch.cuda.is_available()`).
- Regression tests in `tests/test_autobatching.py`: both OOM wordings are caught, and a genuine
  non-OOM error still propagates.

## Candidates evaluated (≥5, tested in parallel on real GPUs)

*OOM-detection logic* (GPU-free harness driving the real function):

| # | Approach | Verdict |
|---|---|---|
| current | as-shipped | reproduces bug (misses Warp OOM) |
| A | broaden default list + `any()`/raise-outside-loop | **works** — chosen |
| B | case-insensitive `"out of memory"` / `"failed to allocate"` | works, broader match than needed |
| C | match by exception *type* + substring | works, extra surface for no gain |
| D | shared `_is_oom_error()` helper | works, more than minimal |
| E | structural fix only, keep narrow default | fails (default still misses Warp) — proves broadening is required |
| F | 3-substring default | works, `"out of memory"` makes `"CUDA out of memory"` redundant |

*Memory-handoff placement* (full-library end-to-end on real GPUs; all include fix #1):

| Variant | Where cache is released | Result |
|---|---|---|
| V0 baseline | nowhere | **FAIL** (`Failed to allocate 2556 bytes`) |
| V1 measure_finally | after every probe forward | **FAIL** (later allocations refill cache) |
| V3 oom_except | only in the OOM catch | **FAIL** |
| V2 post_estimate | before first real batch in `_get_first_batch` | SUCCESS (InFlight-only) |
| **V4 estimate_end** | **end of shared `estimate_max_memory_scaler`** | **SUCCESS — chosen** (covers both batchers) |
| V6 empty_cache only | end of estimator, no sync/gc | SUCCESS (`gc.collect()` unnecessary) |
| V7 sync + empty_cache | end of estimator | SUCCESS (chosen wording) |

Chosen = **A + V4/V7**: most minimal placement that (a) covers both autobatchers via the shared
estimator and (b) needs no `gc.collect()`.

## Validation

- `pytest tests/test_autobatching.py` → **38 passed** (incl. new regression tests).
- orb-v3-conservative, full 4272-structure library, clean dedicated GPU, run **to completion**:
  - **with fix → `ok: 4272/4272`** (mem-est 27.4 s, opt 553 s, 0 failures).
  - identical code with the memory-release disabled (V0 baseline) → `Failed to allocate 2556 bytes`.
  - This A/B (same process-start, cleanup on vs off) is the causal proof for defect #2;
    the GPU-free harness + crash-location shift is the proof for defect #1.

**Note on reproducibility:** the crash is allocator-contention-sensitive. It reproduces
**reliably on a clean/dedicated GPU** (the benchmark's real scenario — it targets an idle GPU),
but a long multi-model run on a *contended* GPU was observed to slip past it once. The fix makes
orb-v3-conservative deterministic regardless of residual allocator state.

## Scope / still open — CONFIRMED
This PR fully fixes **orb-v3-conservative** (the model that crashes on a clean GPU).

**UMA is a separate, confirmed bug this PR does NOT fix.** A clean-GPU run of uma-s-1p1 on the
full 4272-library (with this fix applied, GPU otherwise idle) got deep into optimization and then
died with a genuine PyTorch **self-fragmentation** OOM:
`Tried to allocate 106.00 MiB … 21.85 GiB allocated by PyTorch, 1.51 GiB reserved but unallocated`.
Single process, whole GPU to itself — this is the autobatcher **over-committing at
`max_memory_padding=0.95`** with D3 on, not allocator contention and not external interference.
`empty_cache` (this PR) does not help it. Per prior investigation there is **no clean one-line
torch-sim root-cause fix**; the obvious levers (lower padding, `expandable_segments`) are
workarounds and don't universally hold. Recommend treating UMA fragmentation as separate work.

## Files changed (torch-sim only)
- `torch_sim/autobatching.py`
- `tests/test_autobatching.py`
