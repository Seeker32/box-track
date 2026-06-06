# Goal-Driven Workflow Protocol

Only activate this protocol when the user explicitly invokes a **goal-driven task** (e.g., "use goal mode", "用 goal 模式", or explicitly calls `goal-contract-writer`). For all other conversations, ignore this protocol and respond normally.

## When Activated

The user says something like:
- "用 goal 模式做 X"
- "use goal mode for X"
- "先定义目标再做 X"
- Explicitly calls `goal-contract-writer`

## Protocol

### Phase 1: Define

1. Call `goal-contract-writer` to produce a `Goal Contract`.
2. Show the contract to the user for confirmation before proceeding.

### Phase 2: Execute with Tracking

Once the user confirms the contract:

1. Initialize `goal-progress-tracker` with checkpoint state `none yet` and current focus.
2. Execute toward the goal. After each meaningful step:
   - Collect what happened: completed steps, attempted paths, evidence, blockers
   - Call `goal-progress-tracker` to update the log
3. When a blocker appears, call `goal-progress-tracker` with the blocker state.

### Phase 3: Verify Before Completion

Before claiming the goal is done — or whenever the user asks about status:

1. Gather current contract, progress, evidence, and blockers.
2. Dispatch `goal-contract-verifier` as a fresh subagent.
3. Respect the verdict:
   - `pass` + `complete` → report done
   - `pass` + `blocked` → report blocked, don't claim done
   - `revise contract` → go back to Phase 1
   - `escalate` → report to user, stop execution

### Artifact Storage

Persist artifacts to these file paths so they survive between sessions:

| Artifact | Path | When |
|---|---|---|
| Goal Contract | `.goal/contract.yaml` | After user confirms |
| Goal Progress Log | `.goal/progress.md` | After each tracker update |
| Verifier Verdict | `.goal/verdict-latest.yaml` | After each verification |

Always read the existing `progress.md` at the start of a session to resume from the last checkpoint.

### Automatic Checkpoints

During execution, automatically trigger a tracker update (and verifier if blocked) at:

- A key sub-goal is completed
- Execution path changes direction
- A new blocker appears
- Blocker state changes
- The session is about to hand off or stop