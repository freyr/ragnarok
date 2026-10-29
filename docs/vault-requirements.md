# Vault requirements

ragnarok works on any folder of Markdown notes. This page tells how a vault must look to get the most from
it. The requirements are in four levels, in order of effect. Each item names the ragnarok feature that uses
it. A vault that meets only part of the list still works, but with less precision.

You do not have to move your notes to meet most items. See
[What you can change without a new folder structure](#what-you-can-change-without-a-new-folder-structure).

## Level 1: text search

These items help each search, in every vault layout. The text search (`search`) uses the meaning of the
text and its words. It needs no frontmatter and no project folders.

1. **Markdown notes.** ragnarok indexes only `.md` files. It does not find content in PDF, CSV or TXT files,
   or in images.
2. **Headings in long notes.** ragnarok cuts each note into chunks at its headings. It joins sections with
   fewer than `min_words` (default 80) words and cuts sections with more than `max_words` (default 300)
   words. A good section has one subject. A long note with no headings gives chunks that mix subjects, and
   the search finds them less precisely.
3. **A clear title for each note.** Name the file by its subject, for example `Boiler controller setup.md`,
   not `Untitled 3.md`. The title is part of the search text of each chunk.
4. **English text.** The embedding model and the tests of ragnarok use English. The search finds notes in
   a different language less reliably.
5. **No secrets in notes.** ragnarok removes literal passwords, keys and tokens from the chunks before it
   stores them, but this is only a safety net. Keep credentials in a password manager.

## Level 2: project features

`find_projects`, `list_project` and the `project` filter of `search` use these items. They help most when a
prompt continues earlier work but does not name it.

6. **One folder for each project.** Keep all notes of one piece of work in one folder.
7. **A marker file in each project folder.** A folder that holds a file named in `project_markers` (default
   `DRIVERS.md` and `SUMMARY.md`) is a project folder, and the folder name is the project name. Without a
   marker, the path rules (`project_depth`, `project_parents`, `project_fallback`) find the project from the
   folder level.
8. **A `SUMMARY.md` with the state of the project.** `find_projects` and `list_project` show its
   frontmatter keys `stage`, `state` and `next`, and its `description`. Without a `description`, they show the
   `description` of `DRIVERS.md` or the first paragraph of `SUMMARY.md`. This file must have the exact name
   `SUMMARY.md`; the `project_markers` setting does not change it.
9. **A stable project name.** Use one short name, for example `db-resize`. A note outside the project folder
   can join the project with `project: db-resize` in its frontmatter.

## Level 3: filters and ranks

10. **A small frontmatter in each note.** A frontmatter value always wins over the path rules. Each key is
    optional.

    ```yaml
    ---
    project: db-resize
    area: devops
    type: research
    status: active
    description: Measured CPU load of the database servers before the resize.
    jira: [OPS-77]
    ---
    ```

    - `description`: one sentence about the subject of the note. ragnarok stores it as a separate, short
      chunk, which gives the search a clear target. Of the six keys, it adds the most.
    - `type`: one of `drivers`, `summary`, `research`, `spec`, `plan`, `brainstorm`, `review`, `decision`,
      `guide`, `reference`, `meeting`, `log`, `template` or `note`.
    - `status`: only `archived` changes the behavior. Each other value counts as `active`. A note in an
      `_archive` folder, or in a project whose `SUMMARY.md` has `state: done` or `state: abandoned`, is also `archived`.
    - `jira`: ticket keys. ragnarok also finds keys in the text when their prefix is in `jira_prefixes`.
11. **File names and folders that give the type.** When a note has no `type` key, ragnarok uses these fixed
    rules:
    - the file names `DRIVERS.md` and `SUMMARY.md`;
    - the file name prefixes `research-`, `spec-` and `plan-`;
    - the folder names `plans`, `research`, `specs`, `adrs`, `decisions`, `brainstorms`, `reviews`,
      `solutions`, `todos` and `99-Templates`.
12. **One area for each top folder.** For example `Engineering/` or `Clients/`. By default, the first folder
    of a note gives its area. The `area` rule of `[vault_rules.<vault>]` can change this.
13. **An archive folder.** Move finished work into a folder named in `archive_folder` (default `_archive`),
    or set `status: archived`. The search still finds these notes, and marks them as archived.
14. **Wiki-links between related notes.** `open_note` shows the links and the backlinks of a note, so a
    related note is one step away. A link goes to the note in the same vault that has the most folders in
    common with the linking note, so a project can have its own `DRIVERS.md` and `SUMMARY.md`.

## Level 4: privacy

15. **Sensitive notes in their own folders.** `private_patterns` matches paths, not content. Keep private
    notes in a small number of folders that one pattern can cover, for example `"Notes/Journal/**"`. The MCP
    server refuses these notes.

## What you can change without a new folder structure

Items 7, 9, 10, 12, 13 and 15 need only `config.toml` or frontmatter. The repo skill
`.claude/skills/adapt-vault/` does this work with you (see "Adapting to your vaults" in the
[README](../README.md#adapting-to-your-vaults)). Its backfill writes the frontmatter only after you examine
the proposal. Items 6 and 8 are a change of habit. They matter most for new work.

Use `ragnarok inspect <note path>` to see the metadata that ragnarok gives a note, and `ragnarok status` to
see the frontmatter coverage of each vault.
