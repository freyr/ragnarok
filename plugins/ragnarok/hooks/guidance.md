## Finding notes: ragnarok first

The `ragnarok` MCP server indexes the Obsidian vaults of this user. Use it before Grep or Glob whenever
you look for notes, earlier work or context:

- `find_projects`: the prompt continues or resembles earlier work and names no project. Then call
  `list_project` on the best match (look at the top 3).
- `search`: facts, decisions, how something was done. Filters: vault, project, area, type, archived.
- `open_note`: read a note or one section, with its links and backlinks. A long note gives its outline
  first; open the section you need by heading.
- `open_projects`: which projects are in progress or paused ("what am I working on", "open projects").
- `find_by_jira`: a ticket key such as `ABC-123`.

Notes in a project whose SUMMARY.md has `state: superseded` or `state: abandoned`, or under an archive
folder, are archived. A `done` project is current knowledge, not archived. Include archived notes when
the question is about earlier work.

The tools are deferred. Load them in one call before the first use:
`ToolSearch` with `select:mcp__plugin_ragnarok_ragnarok__search,mcp__plugin_ragnarok_ragnarok__find_projects,mcp__plugin_ragnarok_ragnarok__list_project,mcp__plugin_ragnarok_ragnarok__open_projects,mcp__plugin_ragnarok_ragnarok__open_note,mcp__plugin_ragnarok_ragnarok__find_by_jira`.

If the tools are absent, fall back to Grep and say so; the `ragnarok-setup` skill sets ragnarok up.
Results can be unrelated: read the text before you rely on it, and cite note paths. Private notes are
refused by design; do not work around the refusal with Read or Grep.
