# Lessons

A distilled reasoning library, not a log. `PROJECT_PLAN.md`'s Log records
*what happened, in order* — this file records *what would help on a
different task*, retrievable by relevance instead of by reading
chronologically.

Inspired by the ReasoningBank pattern (Ouyang et al., "ReasoningBank:
Scaling Agent Self-Evolving with Reasoning Memory," arXiv:2509.25140) and
the post-run half of what runtime tools like ReasonBlocks do — extract a
reusable lesson after a task, retrieve the relevant one before the next.
This file replicates the distill-and-retrieve idea; it can't replicate
mid-run interception, which needs a runtime wrapper, not a markdown file.

## What belongs here

A lesson is **generalizable** — it should plausibly help on a task in a
*different* part of the codebase, or even a different project entirely.
If a note only makes sense re-reading this exact task's diff, it belongs
in `PROJECT_PLAN.md`'s Log, not here.

Good lesson: "HF's `Cache.update()` must return `(k, v)` for the current
layer only, not the accumulated cache — easy to get backwards."

Not a lesson (belongs in the Log instead): "Implemented `SimpleKVCache` in
`minigen/cache.py`."

## Format

Each entry: a short title, then 2-4 lines — what happened, what the
generalizable takeaway is, and where it applies. Keep entries short; a
lesson that takes a paragraph to state usually hasn't been distilled
enough yet.

Order: newest first, so skimming from the top surfaces recent lessons
first without needing to search.

Retrieval, per task: skim titles top-to-bottom, pull in anything genuinely
relevant to the current task's domain. Don't inject more than 1-2 lessons
into a task's context — per the ReasoningBank ablation this is based on,
retrieving more hurts more than it helps (relevance beats volume). If
nothing here is relevant to the current task, that's the normal case, not
a gap to fill.

## Separating Control-Plane Latency from Physical Recovery Wall-Clock Time
- What happened: Measuring only internal `reschedule_job()` execution time ($0.496\text{ ms}$) hid physical failure detection, subprocess spawning, and PyTorch CUDA re-initialization delays.
- Takeaway: Benchmark recovery metrics must explicitly decompose control-plane state transition overhead from physical process termination detection, process spawning, and GPU runtime context re-initialization ($129\text{ ms}$).
- Applies to: Distributed fault tolerance, system recovery benchmarks, and latency profiling.

## FastAPI class dependency `__call__` signature parameters
- What happened: Adding extra optional parameters to `APIKeyAuth.__call__` caused FastAPI parameter inspection to treat them as required HTTP body schema parameters, triggering HTTP 422 errors on POST endpoints.
- Takeaway: Class instance dependencies in FastAPI should take only `request: Request` and inspect `request.headers` directly to avoid FastAPI body parameter resolution conflicts.
- Applies to: FastAPI REST endpoints, custom security dependencies, and authentication middleware.

## Node metadata models vs dynamic resource allocation tracking
- What happened: Decrementing `WorkerInfo.total_cpus` down to 0 during simulation triggered a Pydantic `ge=1` validation error.
- Takeaway: Node metadata dataclasses model static hardware capacity, not remaining capacity. Keep static capacity fixed and track remaining allocation separately.
- Applies to: Cluster state modeling, benchmark simulators, and resource tracking.

