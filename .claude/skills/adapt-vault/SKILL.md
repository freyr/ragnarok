---
name: adapt-vault
description: Use when the person asks to adapt, tune or configure ragnarok for their own Obsidian vaults, when search results show the wrong project or area for notes, or right after `ragnarok init` or `./setup.sh` on a new computer. Walks through survey, path rules, private folders, an optional frontmatter backfill, a small evaluation set and the retrieval guidance in ~/.claude/CLAUDE.md. Not for normal searching of notes.
---

# Adapt ragnarok to the vaults of this person

The default path rules expect `<vault>/<area>/<project>/note.md`. Most vaults look different. This skill
changes `config.toml` until each note gets the correct project and area, and optionally adds missing
frontmatter keys. Work in the repo checkout and run every command as `uv run ragnarok ...`. The paths
below assume the default `data_dir = "data"`.

## Hard rules

- Never move, rename or delete a note or a folder in a vault. Change only `config.toml` and frontmatter keys.
- The only command that writes to a vault is `ragnarok backfill apply`. It makes a backup archive first and
  only adds missing keys. Never edit a note by hand for this work.
- Never run `backfill apply` before the person reviewed the proposal and said yes to that proposal.
- Agree on the private folders with the person (step 2) before any rule work, backfill or export, and
  write the private patterns of a vault before you add it to `vaults`.
- Never open, read or quote a note under a private pattern. Do not use Read or Grep to get around the refusal.
- The repo is public. Never put names, paths, notes or questions of the person into tracked files.
  `config.toml` and `data/` are gitignored; keep all of their data there.
- Ask before each step that takes long (`index --rebuild`) or writes outside the checkout.
- Never create, move or delete a project marker file to fix a project; change the rules instead.

## Step 1: survey each vault

```bash
uv run ragnarok survey <vault>          # plain text; --json for one JSON object, --top N for longer lists
```

`<vault>` is a folder name under `root`. It need not be in `vaults` yet. The survey prints folder names,
file names, frontmatter keys and counts, never note text, and lists no value of a private note. Map the
sections to the configuration:

| Survey section | Decides |
|---|---|
| notes by folder depth | `project_depth`, `project_fallback`: at which folder level projects sit |
| folders at level 1 | `area` (`"folder"` if level-1 folders are subjects, else one fixed area), `ignore_folders` (level-1 folders with no subject, for example `assets`, `templates`) |
| folders at level 2 / 3 | `project_parents`: a folder whose children are projects, for example `"Clients/Projects"` |
| frontmatter keys, values of project/area/type/status | how much the frontmatter already gives; frontmatter always wins over path rules |
| file names in 3 or more folders, configured project_markers | `project_markers`: a file name that marks a project folder (for example `README.md`, `index.md`) |
| archive folder names | `archive_folder`: one exact folder name; notes in it get status archived |
| ticket key prefixes | `jira_prefixes` (global, all vaults) or `[vault_rules.<vault>].jira_prefixes` (one vault) |
| frontmatter that cannot be parsed | notes the person must correct by hand; list them for the person |

Show the person a short summary for each vault and your proposed rules before you write them.

## Step 2: agree on the private folders first

Do this before any rule work. `ragnarok init` already indexed every vault in `vaults`, and the MCP server
returns each indexed note that no private pattern covers.

Ask the person which folders hold private notes (journal, health, finance, HR, clients under NDA).
Propose candidates from the survey folder names; do not decide alone. Write them as globs on the path
relative to `root`. The match is case-sensitive, and `*` also matches `/`:

```toml
private_patterns = ["Home/Journal/**", "Home/Finance/**", "Work/People/*"]
allow_private = false
```

Private notes stay in the index, but the server never returns them, the export leaves them out, and the
backfill writes no description for them. Only the person can set `allow_private = true`; never set it.
Check one or two paths with `ragnarok inspect` (`"sensitivity": "private"`), then run
`uv run ragnarok index` and check `private_notes` in `ragnarok status`. The MCP server applies the new
patterns at its next call, with no restart. See "How config changes reach the index" in step 3.

## Step 3: write the path rules and check samples

Edit `config.toml` (copy keys from `config.example.toml`). Global keys: `vaults`, `project_markers`,
`archive_folder`, `jira_prefixes`, `private_patterns`, `eval_exclude`. Put every global key above the
first `[vault_rules.*]` table: TOML puts a key below a table header into that table, and the config then
fails to load. One table for each vault that differs from the defaults:

```toml
project_markers = ["DRIVERS.md", "SUMMARY.md", "README.md"]
archive_folder = "Archive"
jira_prefixes = ["OPS"]

[vault_rules.Clients]
area = "folder"               # the first folder, lower case, is the area; any other value is the area of all notes
area_default = "clients"      # area of notes at the top of the vault or in an ignored folder; "" = vault name
project_depth = 2             # without a marker, the folder at this level names the project
project_fallback = true       # fewer folders: the first folder is the project; false: "none"
project_parents = ["Active/Projects"]   # children of these folders are projects
ignore_folders = ["attachments"]        # first-level folders with no area and no project
jira_prefixes = ["CORE"]      # ticket prefixes only in this vault
```

How a project is picked, in this order: frontmatter `project`; the deepest folder (below the vault) that
holds a file from `project_markers`; a folder under a `project_parents` entry; the folder at
`project_depth`; the fallback.

Check a sample of 8 to 15 notes from different folders and depths (top of the vault, deep folders,
archive, ignored folders). Paths are relative to `root`, with the vault name first:

```bash
uv run ragnarok inspect Clients/Active/Projects/acme-portal/kickoff.md Clients/notes.md
```

Each line is JSON with `project`, `area`, `type`, `status`, `sensitivity`, `jira` and `sources` (where
each value came from: `frontmatter`, `frontmatter-map`, `path` or `default`). Show the person a compact
table. Change the rules and inspect again until the person agrees the values are right. Pick sample
paths from the survey folder names; do not open private notes.

`inspect` reads the files directly, so the rules need no index update while you iterate. Run
`uv run ragnarok index` once when the person agrees the values are right.

### Add a vault to `vaults`

The MCP server reads a changed `config.toml` at its next call. A vault added to `vaults` is then
indexed and searchable at once, and that one tool call embeds the whole vault. So, in this order:

1. Agree the private patterns of the new vault with the person (step 2) and write them first.
2. Add the vault to `vaults`.
3. Run `uv run ragnarok index` at once from the CLI, so that no MCP call has to embed the new vault.

### How config changes reach the index

The index stores a hash of the rule values (`root`, `vaults`, `skip_dirs`, `private_patterns`,
`project_markers`, `archive_folder`, `jira_prefixes`, `vault_rules`). When the hash changes, the next
`ragnarok index`, or the next MCP tool call, derives the metadata of all notes again. Unchanged chunks
keep their vectors, so this is much faster than a rebuild.

```bash
uv run ragnarok index            # after a rule change, the first run reads all notes
uv run ragnarok status           # notes, private notes, rules_hash, frontmatter coverage for each vault
```

The running MCP server reads `config.toml` again before a call when the file changed. A new rule or
private pattern applies at the next tool call; no restart of Claude Code is needed.

Use `ragnarok index --rebuild` (it embeds every note again, approximately 12 minutes for 1,500 notes;
ask first) only after a change of `model`, `dimension`, `min_words` or `max_words`, or in this case:
a project marker file (for example `SUMMARY.md`) added to or removed from a folder does not change
the hash, so the other notes in that folder keep their old project. Touching those notes does not help
(their content hash is the same). Run `ragnarok index --rebuild`.

## Step 4: backfill frontmatter (only if the person agrees)

The backfill adds the missing keys `project`, `area`, `type`, `status` and `description`. Explain that it
writes to the notes, with a backup archive first. Start with the smallest vault (fewest notes in the
survey). The vault must be in `vaults`.

1. Propose:
   ```bash
   uv run ragnarok backfill propose <vault>
   ```
   It writes `data/proposals/backfill-<vault>.yaml` and prints the row count, uncertain rows and
   private rows. A note it reports as "correct its frontmatter by hand" has frontmatter that cannot be
   extended safely; give the list to the person to fix by hand, then propose again.
2. Export:
   ```bash
   uv run ragnarok backfill export <vault>
   ```
   It writes `data/proposals/describe-<vault>.json`: `instructions`, `types` and `rows`, each row with
   `path`, `title`, `type` and a redacted `excerpt` (private rows are left out).
3. Write descriptions. For more than about 40 rows, split `rows` into groups of 30 to 50 and give each
   group to a subagent. Each writes one JSON file in `data/proposals/`, for example
   `data/proposals/desc-<vault>-01.json`:
   ```json
   {"Clients/Active/Projects/acme-portal/kickoff.md": {"description": "Kickoff notes for the Acme portal rebuild: scope, team and first milestones.", "type": "meeting"}}
   ```
   Rules for each description: English, one sentence, 25 words or fewer, from the title and excerpt
   only (do not open the note), no credentials or personal data. `type` is optional and must be one of
   `types`; it replaces only a type whose source is `default`.
4. Merge:
   ```bash
   uv run ragnarok backfill merge <vault> data/proposals/desc-<vault>-*.json
   ```
   It prints merged, missing, too long, redacted and unknown counts. Rewrite the too-long and missing
   rows and merge again; merge can run many times.
5. Review by the person. Open `data/proposals/backfill-<vault>.yaml` for the person and explain the row
   fields:
   - `path`: the note.
   - `approve`: `true` writes this row; set `false` to skip it.
   - `uncertain`: `true` when the project was guessed from the file name (`source.project: match`);
     check these first.
   - `private`: a private note; it gets no description.
   - `set`: the keys and values that will be added. The person can edit any value.
   - `source`: where each value came from: `path` (path rules), `frontmatter`, `default`, `match`
     (file name guess) or `claude` (type from the descriptions).
   - `hash`: the note content at propose time. A note changed after that is skipped at apply.
   Wait for an explicit yes for this proposal.
6. Apply:
   ```bash
   uv run ragnarok backfill apply <vault>
   ```
   It prints the archive path in `data/backups/`, the notes written and the keys added, and each skipped
   note. Give the archive path to the person. Then run `uv run ragnarok index` (the changed notes are
   read again; no rebuild is needed). Repeat for the next vault.

## Step 5: evaluation set

Write 15 to 20 questions from the real notes of the person, so later changes can be measured. Read only
public notes. Put the files in `data/eval/sets/` (gitignored; the default is `<data_dir>/eval/sets`).
Each file is one JSON list.

`data/eval/sets/draft-long.json`, fact questions (12 to 15). `expected` holds the note paths that
answer the question, relative to `root`. `type` is optional; a type that starts with `synthesis` uses
the top 10, any other the top 5:

```json
[
  {"id": "q01", "question": "Which database did we choose for the Acme portal and why?", "type": "fact",
   "expected": ["Clients/Active/Projects/acme-portal/decision-database.md"]},
  {"id": "q02", "question": "What were all the problems with the home heating last winter?", "type": "synthesis",
   "expected": ["Home/Heating/log-2025.md", "Home/Heating/boiler-service.md"]}
]
```

`data/eval/sets/hard.json`, continuation questions (3 to 5): a prompt that continues a project without
its name. Every item has `subtype`; `expected` holds project folder paths, the first is the main one:

```json
[
  {"id": "h01", "type": "continuation", "subtype": "no-name", "question": "Let's continue the portal rebuild, where were we?",
   "expected": ["Clients/Active/Projects/acme-portal"]}
]
```

Use the words a person would type, not the exact words of the note. A question whose expected notes are
private or not indexed is dropped. If a note quotes the questions (for example a planning note about
this work), add its path prefix to `eval_exclude` (a prefix, not a glob), above the first
`[vault_rules.*]` table:

```toml
eval_exclude = ["Work/Meta/search-eval"]
```

Run it and record the output for the person (results are also written to `data/eval/results/`):

```bash
uv run ragnarok eval mechanical
```

It compares grep, keyword, dense and hybrid search. Hybrid should be at least as good as grep. If a
project scores low, inspect its notes again (step 3).

## Step 6: retrieval guidance in ~/.claude/CLAUDE.md

Claude Code loads MCP tools on demand, so Claude uses them only when told to. Ask the person before you
edit `~/.claude/CLAUDE.md`, show the block, then add it (adjust the vault names):

```markdown
## Finding notes: ragnarok first

The `ragnarok` MCP server indexes my Obsidian vaults (Work, Home). Use it before Grep or Glob whenever
you look for notes, earlier work or context:

- `find_projects`: the prompt continues or resembles earlier work and names no project. Then call
  `list_project` on the best match (look at the top 3).
- `search`: facts, decisions, how something was done. Filters: vault, project, area, type.
- `open_note`: read a note or one section, with its links and backlinks.
- `find_by_jira`: a ticket key such as `OPS-77`.

The tools are deferred. Load them in one call before the first use:
`ToolSearch` with `select:mcp__ragnarok__search,mcp__ragnarok__find_projects,mcp__ragnarok__list_project,mcp__ragnarok__open_note,mcp__ragnarok__find_by_jira`.

If the tools are absent (for example on a computer without ragnarok), fall back to Grep and say so.
Results can be unrelated: read the text before you rely on it, and cite note paths. Private notes are
refused by design; do not work around the refusal with Read or Grep.
```

## Done when

- [ ] `ragnarok inspect` on a sample from each vault shows the project and area the person expects.
- [ ] Private patterns are agreed, written, and checked with `inspect`.
- [ ] `ragnarok index` ran after the last configuration change (`ragnarok status` shows all vaults).
- [ ] The backfill ran for each vault the person chose, or the person declined it.
- [ ] `ragnarok eval mechanical` ran on 15 to 20 questions; the result is shown to the person.
- [ ] The guidance block is in `~/.claude/CLAUDE.md`.
- [ ] In a new session, a prompt that continues earlier work without a project name makes Claude call
      `find_projects`.
