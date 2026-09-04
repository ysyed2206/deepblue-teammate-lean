@"C:\Users\mohib\aichessathon-starter/" DEEP BLUE — AUTONOMOUS MAIN-REPO DEVELOPMENT

You are working directly inside my live repository:

C:\Users\mohib\aichessathon-starter

I have made and verified external backups immediately before this session.

You MAY modify this repository to develop and test Deep Blue.

However there is ONE ABSOLUTE FILE BOUNDARY:

==================================================
NNUE IS CURRENTLY RUNNING — DO NOT TOUCH IT
==================================================

`nnue_lab/**` IS STRICTLY READ-ONLY.

You may inspect NNUE files if necessary, but you must NEVER:

- modify anything under nnue_lab/
- delete anything under nnue_lab/
- rename anything under nnue_lab/
- format anything under nnue_lab/
- create files under nnue_lab/
- move files into or out of nnue_lab/
- clean caches under nnue_lab/
- alter its environment
- kill or restart NNUE-related processes
- interrupt its downloads/training
- run a Git operation that may overwrite nnue_lab/
- run repo-wide reset/clean/checkout commands that could affect nnue_lab/

Specifically NEVER run:

git reset --hard
git clean -fd
git clean -fdx
git restore .
git checkout -- .
git checkout <branch> -- .
or any equivalent whole-tree destructive operation.

Do not run a repo-wide formatter.

The NNUE worker is independent and must be allowed to continue completely undisturbed.

==================================================
CHAMPION SAFETY
==================================================

The currently accepted classical champion is:

deepblue.fastsearch4

DO NOT overwrite fastsearch4.py.

DO NOT silently make another candidate the champion.

Experiments should normally be created as new variants:

fastsearch18
fastsearch19
...

You may freely modify/add:

deepblue/ new candidate modules
tools/
tests/
harness test infrastructure
other development support files

But do not replace the accepted champion merely because a candidate compiles.

`agent.py`, `champion/`, and competition submission ZIPs should remain untouched during ordinary experimentation.

Only alter integration/packaging files if you reach an explicit final-integration phase and clearly document it first.

Do not modify dependency specifications unless genuinely required:

pyproject.toml
uv.lock

Prefer existing dependencies.

==================================================
DONOR REPOSITORIES
==================================================

Aggressively inspect ACTUAL donor source, not summaries.

Relevant donors:

- Reckless
- Sykora
- Pawnstar
- Coda
- Stockfish 18
- PlentyChess
- Obsidian
- black_numba

Use:

git history
historical tags
git diff
actual source files
tests
commit bodies
SPRT results
tuning files
changelogs

Do not guess what an engine does from its README when source/history can answer it.

You are encouraged to closely inspect third-party implementations so we do not independently recreate known algorithms incorrectly.

However:

DO NOT paste substantial third-party source verbatim into Deep Blue.

Independently implement the learned architecture/formulas/control flow for our Numba engine.

==================================================
RULE: VERIFY EVERY CLAIM AGAINST DEEP BLUE
==================================================

Previous research has already contained at least one material mistake.

Before implementing any claimed "missing" feature:

1. open the actual Deep Blue source
2. verify whether it is genuinely missing
3. open the actual historical donor implementation
4. determine the REAL delta
5. only then implement

Maintain:

DEEPBLUE_AUDIT_CORRECTIONS.md

If previous notes are wrong, record the correction there.

Do not perpetuate a bad assumption just because it appears in a planning document.

==================================================
CURRENT ARCHITECTURE
==================================================

Accepted fastsearch4 includes:

- custom Numba bitboard core
- legal move generation
- make/unmake
- Zobrist
- repetition/history handling
- TT
- killers/basic history
- PVS
- Reverse Futility Pruning
- tactical-only qsearch generation

Validated SEE infrastructure exists separately.

SEE was differential-tested on:

8045 captures
0 exact-value mismatches
0 sign mismatches

==================================================
KNOWN REJECTED EXPERIMENTS
==================================================

Do NOT build new work on rejected variants by default.

NMP v1 / fastsearch5:
paired score 40.6%

LMR v1:
43.8%

raw numeric SEE ordering:
37.5%

combined history Patch07 / fastsearch12:
46.9%

bounded History Patch A:
correct and invariant-safe but standalone structural metrics regressed

Reusable utilities that were independently validated, such as SEE, may be extracted.

==================================================
EXTERNAL BASELINE
==================================================

At 100 ms/move:

vs Stockfish UCI_Elo 1800:
84.4%

vs 2200:
40.6%

vs 2500:
15.6%

64 games vs Stockfish UCI_Elo 2140:

+22 =11 -31
43.0%

This is approximately low-2100 Stockfish-UCI-scale strength at this time control.

This is calibration only.

Do not treat Stockfish UCI_Elo as competition Elo.

==================================================
OBJECTIVE
==================================================

We want maximum legitimate playing strength as quickly as possible.

Do NOT spend all your time producing prose.

Operate as an autonomous:

SOURCE ARCHAEOLOGIST
→ IMPLEMENTER
→ TESTER
→ RED-TEAM AUDITOR
→ PATCH FARM

Continue making progress without waiting for me after every patch.

For every candidate:

1. inspect Deep Blue
2. inspect donor implementation/history
3. write a short implementation spec
4. implement a NEW candidate
5. inspect git diff
6. run correctness tests
7. run structural tests
8. run timed tests
9. red-team your own code against donor + Deep Blue semantics
10. run playing gate if justified
11. record result
12. proceed to next experiment

Never automatically replace fastsearch4.

==================================================
TEST PHILOSOPHY
==================================================

A feature is NOT accepted because:

- Stockfish has it
- Reckless gained Elo from it
- it reduces nodes
- it reaches greater depth
- it is theoretically modern

Promotion requires Deep Blue evidence.

But do not waste 1000 games testing something already structurally broken.

Order:

targeted correctness
regression
invariants/differential
equal-depth
timed
small playing gate
larger paired/self-play test where justified

==================================================
PRIMARY DONOR ROADMAP
==================================================

Historical Reckless rebuild is particularly valuable because features were re-added one at a time with SPRTs.

Approximate donor sequence:

MVV-LVA
TT cutoff
TT move ordering
PVS
RFP
butterfly history
TT semantic improvements
NMP
TT move preservation
depth-zero check handling
LMR
SEE noisy separation
LMP
futility
bad-noisy qsearch skipping
SEE pruning
aspiration
singular
qsearch TT
time management
staged move picker

Use this as evidence, not scripture.

Deep Blue is architecturally different.

==================================================
CURRENT HIGH-PRIORITY DEVELOPMENT
==================================================

First finish evaluating existing new variants 14, 15 and 17.

After that, likely priorities are:

1. TT inside qsearch
2. donor-exact history decomposition
3. SEE good/bad noisy partition + qsearch skip
4. LMR only after ordering is healthy
5. conservative LMP
6. correctly audited NMP v2
7. RFP quiet-TT guard A/B
8. aspiration / correction-history / staged picker if time permits

Do not spend meaningful time tuning killers.

==================================================
CRITICAL SEMANTICS
==================================================

Preserve all Deep Blue correctness around:

- mate before 50-move draw
- claimable draws
- repetition history
- legal-EP canonical hashing
- pinned pseudo-EP not affecting canonical hash
- mate-score TT normalization
- TT bounds
- stopped searches
- MAX_PLY
- make/unmake
- promotion
- castling
- en passant
- exact Numba integer semantics

Do not store history-dependent repetition results as context-free exact TT facts.

Do not accidentally make rule50-sensitive TT entries universally reusable.

==================================================
NUMBA
==================================================

Watch for:

- object-mode fallback
- dynamic allocation in recursive hot path
- type widening
- uint64 boxing
- huge arrays harming cache
- Python calls from njit
- signed integer floor/truncation differences
- compile-time explosions

Native FastEngine4 construction was measured around 2.31 seconds.

Competition initialization allowance is 60 seconds.

Warn if a candidate cold start approaches 45 seconds.

==================================================
OUTPUT
==================================================

Maintain at repo root:

CLAUDE_REMOTE_RESULTS.md

For every experiment record:

ID
parent
donor/reference
exact code delta
correctness result
equal-depth result
timed result
playing result
bugs found
deviations from donor
recommendation:
KEEP / REJECT / HOLD / NEEDS USER GATE

Maintain:

DEEPBLUE_AUDIT_CORRECTIONS.md

Never hide negative results.

If a patch fails, preserve the evidence, return to the accepted parent, and continue with another independent experiment.

Proceed autonomously.
CURRENT SESSION STATE — READ THIS BEFORE DOING NEW WORK

Three new candidates have just been installed and structurally tested.

==================================================
FASTSEARCH14 — P1a
==================================================

Purpose:

Preserve an existing same-key TT move when a new TT write has best_move == 0.

Regression:

32/32 PASS

Equal-depth 7 versus fastsearch4:

start:
285058 → 285058

kiwipete:
574210 → 574210

closed:
463365 → 463365

tactical:
93517 → 93517

endgame:
49925 → 49925

Every best move and score identical.

Interpretation:

The patch is completely node-identical on the standard fixed-depth suite.

Therefore it appears structurally inert on these searches.

Do NOT assume donor +38.9 Elo transfers.

Determine whether:
- the condition is simply rare,
- our TT replacement semantics differ,
- or the patch is genuinely irrelevant.

It still needs timed/playing evidence before final rejection, but it is LOW priority unless instrumentation demonstrates the mechanism actually fires.

==================================================
FASTSEARCH15 — P1b
==================================================

Purpose:

Allow TT move ordering at PV nodes but restrict TT score/bound cutoffs to non-PV nodes.

Regression:

32/32 PASS

Equal-depth 7:

start:
285058 → 285264

kiwipete:
574210 → 574360

closed:
463365 → 463411

tactical:
93517 → 94188

endgame:
49925 → 49961

Median node ratio:
1.001

Best moves and scores identical at depth 7 on all five.

Interpretation:

This is a very small structural change, not a tree explosion.

It is worth timed + playing testing.

Audit the exact PV classification before trusting it.

==================================================
FASTSEARCH17 — P4
==================================================

Purpose:

Do not enter qsearch at depth zero while in check; floor such nodes to depth 1.

Also adds MAX_PLY safety guard.

Regression:

32/32 PASS

Equal-depth 7:

start:
285058 → 284941      -0.04%

kiwipete:
574210 → 569254      -0.86%

closed:
463365 → 509871     +10.0%

tactical:
93517 → 91046        -2.64%

endgame:
49925 → 42791       -14.29%

Median node ratio:
0.991

Best move and score at depth 7 identical on all five.

Interpretation:

Mixed but interesting structural effect.

No catastrophic blow-up.

Definitely continue timed + playing testing.

==================================================
TIMING WARNING
==================================================

Do not infer much from the single fixed-depth elapsed-time values.

They varied heavily with warmup / OS effects.

Use node counts for fixed-depth structure.

Run proper repeated timed tests.

==================================================
NEXT ACTIONS
==================================================

FIRST:

run repeated timed comparison of:

fastsearch4
fastsearch14
fastsearch15
fastsearch17

at approximately 1 second.

SECOND:

run small paired 16-game gates:

fastsearch14 vs fastsearch4
fastsearch15 vs fastsearch4
fastsearch17 vs fastsearch4

80 ms/move, 8 paired openings.

Do not overinterpret 16 games.

They are rejection/smoke gates only.

If 14 remains completely inert, instrument its mechanism before spending a large test on it.

If 15 or 17 look sane/positive, prepare them for more rigorous paired testing.

==================================================
IMPORTANT CORRECTION ABOUT NMP
==================================================

A previous donor audit stated that Deep Blue NMP v1 likely failed because it lacked several standard guards.

ACTUAL SOURCE INSPECTION SHOWED THAT CLAIM WAS PARTLY WRONG.

Rejected fastsearch5 already includes at least:

- static_eval >= beta guard
- non-pawn-material / zugzwang-oriented guard
- consecutive-null prevention
- EP-aware null hash handling

Therefore DO NOT implement an NMP v2 based on "adding" those features.

Before touching NMP again:

open fastsearch5
open historical Reckless NMP source
diff the actual algorithms
create NMP_DELTA.md

Likely real areas to inspect include:

- reduction formula / adaptivity
- eval-beta scaling
- exact null depth calculation
- PV/cut-node handling
- mate handling
- return semantics
- verification
- interaction with RFP
- TT interaction

But VERIFY rather than assume.

==================================================
PATCH A
==================================================

Do not install/revive the previous Patch A wholesale.

It used:

MAX_HISTORY 16384
asymmetric malus approximately 5x bonus
extra decay
multiple simultaneous design changes

and failed its structural promotion gate.

Historical donor evidence instead decomposes history changes.

If working on history, test isolated donor-faithful components.

==================================================
NNUE
==================================================

An independent NNUE worker is CURRENTLY RUNNING.

ABSOLUTELY DO NOT TOUCH:

nnue_lab/**

Do not kill its processes.

Do not clean its caches.

Do not run repo-wide Git operations capable of modifying that directory.

Do not try to integrate NNUE while it is running.

It is expected to deliver independently.

==================================================
DEVELOPMENT MODE
==================================================

You may now modify the MAIN REPOSITORY outside nnue_lab.

You do not need to create a shadow copy.

A full external backup exists.

However:

- keep fastsearch4 untouched
- create new variants
- do not silently promote
- keep agent.py/champion/submission artifacts untouched until explicit final integration
- audit your changes carefully
- continue autonomously while I am away

Proceed.
Do not sit idle while playing tests run. Pipeline work.

While candidate N is playing, audit/build candidate N+1.

You may prepare several independent candidate variants ahead of me.

Do not stack a rejected candidate into later work.

If dependency requires a not-yet-accepted parent, preserve both:
- independent candidate where possible
- cumulative candidate explicitly labelled as dependent

When I return I want multiple tested candidates, not one result and a question asking what to do next.
okay here is an example, but if you have a better appraoch use urs: (Yes. I checked Anthropic’s current guidance, and for this exact job I would use Sonnet, not leave it on Opus.
Anthropic currently says Sonnet is the default/right choice for most coding because it is fast and quota-efficient; Opus is for the genuinely hard architectural/debugging decisions and consumes meaningfully more quota. They explicitly recommend the pattern “plan with Opus, execute with Sonnet”, and Claude Code has `/model opusplan` for exactly that. [Anthropic Help Center](https://support.claude.com/en/articles/14552983-models-usage-and-limits-in-claude-code?utm_source=chatgpt.com)
So my choice is:
Main autonomous session: Sonnet 5, High effort.
Architecture/really nasty debugging: Opus via planning/advisor/subagent, not as the full-time executor.
If `/model` shows different names on your account, that command is the source of truth; current Anthropic docs list both Claude Sonnet 5 and Opus 5 as active. [Claude Docs](https://docs.anthropic.com/en/docs/about-claude/model-deprecations?utm_source=chatgpt.com)
Why Sonnet wins here
We need perhaps hours of repetitive repo work: open donor source, diff it, implement 30 lines, run regression, inspect logs, make next variant, repeat. Burning Opus on every `rg`, test run and mechanical edit is exactly how you hit the limit again. Anthropic says model choice materially affects usage, and Opus consumes several times more quota per turn than Sonnet. [Anthropic Help Center](https://support.claude.com/en/articles/14552983-models-usage-and-limits-in-claude-code?utm_source=chatgpt.com)
The intelligence bottleneck is only ~20% of this workload:
“Is this donor feature actually compatible with our PVS/TT semantics?”
“Why did this patch regress despite apparently correct implementation?”
“Which of these historical implementations is the right donor?”
That is where Opus earns its cost.
Yes, absolutely use multiple agents
Anthropic’s latest models can orchestrate subagents natively, and their own guidance says to use them for independent parallel workstreams / isolated context while avoiding them for simple sequential tasks. [Claude Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-templates-and-variables?utm_source=chatgpt.com)
But do not let four agents concurrently edit `fastsearch18.py` or the same repo files. That is how we create race-condition garbage.
I would add this final prompt to the Claude session after the prompts I already gave you:
PARALLEL AGENT STRATEGY
Use subagents aggressively where they genuinely reduce wall-clock time.
You are the MAIN COORDINATOR and primary writer.
Default subagent roles:
Agent A — Donor Archaeologist, READ ONLY

* inspect historical Reckless/Sykora/Pawnstar/Coda/Stockfish source
* locate exact commits/implementations relevant to the next candidate
* compare formulas, guards, ordering and dependencies
* report precise findings to coordinator
* do NOT modify Deep Blue

Agent B — Deep Blue Red-Team Auditor, READ ONLY

* inspect the actual parent candidate / fastsearch4
* look for existing implementation of supposedly missing features
* analyze TT/PVS/draw/Zobrist/Numba interactions
* specifically try to falsify the proposed patch before implementation
* update audit findings through the coordinator
* do NOT modify engine source

Agent C — Test / Evidence Analyst, READ ONLY except test-support files when explicitly assigned

* inspect existing regression / benchmark / paired-test infrastructure
* analyze completed candidate results while the main coordinator works on the next patch
* detect noisy or invalid experiments
* design targeted invariant tests
* may add test-only tooling if explicitly delegated and files do not overlap the main agent's edits

Main Coordinator — WRITER

* synthesizes A+B+C
* independently implements the candidate
* owns candidate engine source files
* runs final diff review
* packages / records results

Do not allow multiple agents to write the same file concurrently.
Do not allow any subagent to touch `nnue_lab/**`.
If parallel write work is genuinely worthwhile, give each writer a DISJOINT candidate/file set or separate Git worktree. Never have two agents edit the same candidate.
While long tests run, keep useful work moving:

* donor archaeology for candidate N+1
* red-team candidate N
* analyze candidate N-1 results

But avoid CPU-heavy parallel benchmarks while an Elo/self-play test is running. We have one machine and chess tests need stable CPU allocation.
Prefer parallelism for FILE READING, SOURCE RESEARCH, CODE AUDITING and RESULT ANALYSIS; serialize CPU-heavy benchmarks and edits to shared hot-path code.
Use subagents when tasks are genuinely independent. For a one-file mechanical change, work directly rather than spawning an agent.
Whenever a patch seems unexpectedly positive or negative, have a fresh subagent audit it independently before stacking another feature on top.
<investigate_before_answering> Never speculate about code you have not opened. Read the actual relevant Deep Blue and donor files before making a claim about implementation. </investigate_before_answering>)
Before P2, P3, P5, P6 and P8 implementation, use an Opus planning/review agent if available. Do not use Opus for routine execution.Persist detailed experiment state in CLAUDE_REMOTE_RESULTS.md and periodically re-read that file rather than repeating all previous outputs into chat. Refer to code by path and open selectively. Do not paste giant benchmark logs into the conversation when the full output is already stored on disk.NNUE is independently running. Before a timed benchmark or self-play test, inspect current CPU load. Do not run multiple CPU-heavy chess tests simultaneously. Source reading, code analysis and donor mining may remain parallel. Treat wall-clock/NPS results obtained under heavy NNUE/training load as potentially contaminated; fixed-depth node counts remain usable.