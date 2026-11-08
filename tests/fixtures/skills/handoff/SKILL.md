---
name: handoff
description: Use when state must survive into a session that remembers nothing — a project stage closes, a session ends, context runs low, or the user asks "where are we", "what's the status", or "summarize where we got to". Writes SUMMARY.md in the project folder: stage, gate, next action, what is true now, and what is still unknown. Also use when resuming a project, to read that file back before anything else. Not for a commit message or a chat recap.
---

# Handoff

Write for a reader who has no memory of this conversation and will not read the
transcript. That reader is you, tomorrow.

## The file

One per project: `SUMMARY.md` in the project folder. **Rewrite it whole, every
time.** Never append, never date-stamp a change, never keep a history. It states
what is true now, as if it had always been so.

If the project has none, create it. If it has one, the new version replaces it —
including deleting sentences that are no longer true.

## Shape

```markdown
---
project: <slug>
stage: FRAME | EXPLORE | DESIGN | DELIVER
state: in-progress | paused | done | superseded | abandoned
gate: open | awaiting-approval
language: ASD-STE100
next: <one imperative sentence>
updated: <YYYY-MM-DD>
review: <type> <target> round <n> of <max> — <ledger path>   # optional
---

# <Title>

<What this is and why it exists. Then what is true now: what has been decided,
what has been built, what the evidence says. Name documents as [[wiki-links]]
and code by `path`.>

## Open
<What is unresolved, what is unconfirmed, what would change the answer.>

## Next
<The single next action, specific enough to start without asking a question.>
```

The frontmatter is the machine-readable half. It is how a cold session finds the
stage, and how one grep finds every project waiting on the user. The vault
registry (`## Vaults` in `~/.claude/CLAUDE.md`) gives the vault paths:

```bash
grep -rl "gate: awaiting-approval" <each vault path in the vault registry> --include=SUMMARY.md
```

### `stage` and `state`

`stage` is the phase the work is in. `state` is the overall status of the
project, independent of the phase: a project can be paused or abandoned at any
stage, and a finished project keeps the stage it ended in.

- `in-progress` — being worked on.
- `paused` — stopped on purpose, to resume later. `## Next` says what unblocks it.
- `done` — the goal is met. The body states the result.
- `superseded` — a later project, skill, or document replaces it. Add
  `superseded_by:` with the successor's path; the body says what replaced what.
- `abandoned` — stopped for good, or no longer valid (the approach was dropped,
  the tool or hardware is gone). The body says why.

A `done` project stays where it is. Its notes are current knowledge of the area
it belongs to: a delivered project is part of what the area knows, not history.

Only a project that is `superseded` or `abandoned` moves to
`<Area>/_archive/<slug>/` — ask before the move. ragnarok marks the notes of
both as archived. Pruning an area is a content decision made note by note,
never a folder sweep.

### The `review:` field

Write it only when a `review-loop` is open as the session ends — a round ran and
the loop has not stopped. Delete it once the loop stops. Example:

```yaml
review: plan plan-narrow-skills.md round 2 of 3 — /private/tmp/.../review-plan-ledger.md
```

The ledger lives in the scratchpad, which a later session may not have. When the
ledger path no longer exists, the next session restarts the loop at round 1 on
the current target and says so; it does not invent the earlier rounds.

## Precision test

Check each line against these. A line that fails is noise.

- **Would a stranger have to ask a question to act on this?** Then it is not
  specific enough. "Fix the caching" fails; "add the tenant id to the cache key
  in `src/Cache/KeyBuilder.php:44`" passes.
- **Is it named?** Files by path, projects by slug, people by role, decisions by
  what was decided. Not "the config", "the script", "the meeting".
- **Is the number there?** A measurement without its value or its source is a
  feeling.
- **Does it say what is *not* known?** The most expensive thing to lose is the
  memory of what you were unsure about. Uncertainty is content, not a confession.
- **Would it still be true in a month?** Then it belongs. If it is about the
  conversation — what you tried, in what order, what went wrong on the way — it
  does not.

## Alternatives

Record a rejected option only with the reason it fails a named driver. That is
current truth about the design space. A list of things you thought about is not.

## Language

`SUMMARY.md` is produced documentation, so it is written in ASD-STE100 Simplified
Technical English. The rules and the dictionary are in `~/.claude/STE.md`.

**When you use a domain word that is neither in the ASD-STE100 dictionary nor in
`~/.claude/STE.md`, add it to `~/.claude/STE.md` before you finish.** Give it one
part of speech and one meaning. Nothing else checks this; if you skip it, the
dictionary silently stops matching the documents.

## Length

As long as the state requires, and no longer. Most projects are under a page. A
handoff that restates the spec has replaced the spec with a worse copy — link to
it instead.
