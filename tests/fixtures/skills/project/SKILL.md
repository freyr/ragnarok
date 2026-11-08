---
name: project
description: Use when work needs more than one step or more than one session — "start a project", "I need research into", "figure out how", "look into", "what should we do about", "design a", "I want to know". Also when a folder holds DRIVERS.md or SUMMARY.md, when resuming or closing a project, when asking where one stands, or when naming a stage: FRAME, EXPLORE, DESIGN, DELIVER. Not for a single question, a lookup, or a one-file edit.
---

# Project

## Is this a project?

Decide this first, before creating anything.

- One answer to look up → **not a project**. Answer it.
- One edit, no design decision, one session → **not a project**. Do it.
- Needs a plan, several steps, or a decision that outlives the session → project.

The description above triggers broadly on purpose, so a wrong start is expected.
Exiting here is cheap and correct: say which you think it is, in one sentence, and
carry on with the ordinary answer. Do not create a folder to be safe.

## First move on a real project

0. **Find the project first when the prompt does not name one.** Call the
   `ragnarok` tool `find_projects` with the prompt (load it with ToolSearch if
   it is deferred), look at the top 3, then `list_project` on the best match.
   A new project that resembles earlier work does this too — the earlier
   project's findings and decisions are context. If nothing fits, it is new.
   Fall back to Glob/Grep only when the server fails.
1. Read `SUMMARY.md` in the project folder. **It is the only state.** Frontmatter
   gives the stage and the next action; the prose gives what is true now.
   No `SUMMARY.md` means the project does not exist yet — you are at FRAME.
2. Read `DRIVERS.md` if it exists.
3. If the folder has a `docs/` subfolder, load `stages/convert.md` first.
4. Load exactly one stage file — `stages/frame.md`, `stages/explore.md`,
   `stages/design.md`, `stages/deliver.md` — and nothing else.

Never reconstruct state from the transcript, from file timestamps, or from what
you remember. **If `SUMMARY.md` disagrees with your memory, `SUMMARY.md` wins.**

## The four stages

| Stage | Produces | Ends when |
|---|---|---|
| FRAME | `DRIVERS.md` | the user accepts the drivers |
| EXPLORE | `research-*.md` | the user accepts the findings |
| DESIGN | `spec-*.md` and `plan-*.md` | the user accepts the plan |
| DELIVER | code in the mapped repo, or files in `assets/` when there is no repo | the user accepts the result |

## At a stage boundary

1. Write the stage's artifact.
2. Invoke the `handoff` skill to rewrite `SUMMARY.md`.
3. Report: what you produced, what you assumed, what is still unknown, what the
   next stage would do. Then stop and let the user decide.

**The user drives progression between stages.** Your obligation at a boundary is
the handoff, not a blockade — the handoff is why the boundary exists. Do not
start the next stage's artifact on your own momentum, and do not treat "carry on"
as advance permission for a stage the user has not seen the handoff for.

**A stage may be closed empty, never skipped silently.** If there is nothing to
explore, write one line saying so and write the handoff anyway.

## Where it lives

Design output goes to the vault, code goes to the repo. Implementation output
that is not code and has no repo goes to the project's `assets/` folder. Load
`artifact-routing` to resolve which vault and which repo — do not guess paths.

A project is one folder in a vault area. Documents sit flat at its root; files the
work produces sit in `assets/`:

```
<vault>/<Area>/<slug>/
├── DRIVERS.md              goal, quality attributes, constraints, concerns
├── SUMMARY.md              the only state document
├── research-<question>.md  EXPLORE output
├── spec-<slug>.md          DESIGN — what and why
├── plan-<slug>.md          DESIGN — the ordered steps
└── assets/                 DELIVER output when the project has no repo
```

No `docs/` subfolder. The filename prefix and a `type:` frontmatter key carry
what the folder used to. Filenames carry no date — a date invites a second note
where a rewrite belongs.

`DRIVERS.md` frontmatter carries `repo:`. `none` means the project is standalone,
and its DELIVER output goes to `assets/`.

## The assets folder

Some projects deliver something that is not code and has no repo to hold it: a
deck built in Canva, a part modelled in Onshape, a run sheet for an event. The plan
still produces files — images, slide text, exports, checklists — and those files
need a home that survives the session. `assets/` is that home.

**Why a named folder.** Without one, the output drifts to `~/Downloads`, the
scratchpad, or the project root. The first two are outside the vault: not synced,
lost or unfindable for a cold session, and never where the user looks for project
work. The root belongs to the documents, and deliverables mixed into it bury what
the design says under what the work produced. The `~/Downloads` drift has already
happened once.

**What goes in:**

- Files the deliverable is built from — photos and sketches placed on slides, text
  written to be pasted into an external tool, data a model or a printer reads.
- Files that are the deliverable or its backup — PDF and PPTX exports, STL files,
  printable sheets, run sheets and checklists used on the day.
- Working drafts a plan task hands to the user to act on.

**What stays out:**

- Documents. `research-`, `spec-`, and `plan-` files stay flat at the root, and
  `SUMMARY.md` stays the only state.
- Code, tests, and config — they go to a repo. A vault that permits code in a named
  area (the Private vault's `3dPrints/`) says so in its own `CLAUDE.md`.
- Throwaway intermediates — use the scratchpad.
- Originals the source can supply again. Keep the version the deliverable uses —
  the 10 MB JPEG, not the 200 MB TIFF behind it. The vault syncs every byte.

**Rules:**

- Keep `assets/` flat. Group into subfolders only when a flat list stops being
  readable, and group by what the files serve (a slide, a part), never by date.
- Name each file by what it is — `M31-sketch-Vlasov.jpg`, `task-1-slides-1-and-8.md`
  — with no date in the name.
- Every external file is traceable: a research note or the plan names its source
  and its license.
- The plan's target names `assets/`, and a task that writes there names the files
  its check looks at.
- `SUMMARY.md` names the assets a cold session needs in order to continue. It does
  not inventory the folder.
- Create `assets/` when DELIVER first writes to it, not earlier.

## Language

Every document at the project root is produced documentation, written in
ASD-STE100 Simplified Technical English per `~/.claude/STE.md`. This file and the
stage files are not — they are instructions for a model. Files in `assets/` use
whatever language the deliverable needs, such as Polish slide text.

## Never

- Never create `README.md`, a decision log, `BRIEF.md`, or a per-project
  `CLAUDE.md`. The files in the tree above, and `assets/`, are the whole set.
- Never write code, tests, or config into a vault. `assets/` holds output that is
  not code.
- Never leave project output outside the vault or the repo — not in `~/Downloads`,
  not in the scratchpad.
- Never infer a driver. Drivers are asked for; an unconfirmed target is written
  as `unconfirmed`, not invented.
- Never leave two documents stating the same fact differently. Rewrite, delete
  the contradiction, and do not annotate what changed.
