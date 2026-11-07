# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-11-07

### Added

- Search shows the knowledge-base labels of a note (`class`, `validity`, `as_of`, `superseded_by`) when it has
  them, and leaves out notes with `validity: superseded` or `rejected` unless `include_superseded` is true.
  `list_project` shows `subject` and `class`, with a default `class` from `type`.
- Entity notes (`type: entity`): the `find_entities` tool, the resolution of `subject`, `owner`, `supersedes` and
  `superseded_by`, and a `subject` filter of `search` that applies before the fusion.
- `ragnarok validate [<path>...] [--json]` and the `validate` tool check notes, or a draft before a write,
  against the knowledge-base frontmatter model: the values, the required keys, the links, the reference targets
  and the identity keys. With no path, the command prints the counts of broken links, double identity and notes
  with no error.
- `ragnarok find-collisions <path>|-` and the `find_collisions` tool give at most 10 notes that a draft can
  collide with, from the identity key, the hybrid search, the links of the draft and the `supersedes` targets.
- The `vault_rules` tool gives the path rules of each vault and the new `[write]` table (`known_keys`,
  `systems_of_record`). Both lists are empty by default.
- `ragnarok replay build|score|run` build and score a replay set of the write path, for measurement.

### Changed

- With no filter, the search order is unchanged. An older index gets the new columns from the embedding cache,
  with no new embeddings.

## [1.1.0] - 2026-11-05

### Added

- ragnarok is on PyPI as `ragnarok-mcp` and in the MCP Registry as `io.github.freyr/ragnarok`.
  `uvx ragnarok-mcp serve` runs the
  server with no checkout. The workflow `publish.yml` publishes a tag `v<version>` to both.
- Outside a checkout and with no `RAGNAROK_HOME`, the home folder is `~/.ragnarok`. The package holds
  `config.example.toml`; the checkout has a link to it.
- `ragnarok doctor` and `ragnarok init` accept the server of the Claude Code plugin `ragnarok`
  (`plugin:ragnarok:ragnarok`). A user registration next to the plugin server is a failure, and init removes
  it after it asks. Outside a checkout, init registers `uvx ragnarok-mcp serve`, with `RAGNAROK_HOME` when the home
  folder is not `~/.ragnarok`.

- The `[vault_paths]` table gives the folder of a vault that is not in `root`. The note keys stay
  `<vault>/<path in the vault>`, so the current index stays valid. `ragnarok doctor` names each vault folder
  that does not exist, and `ragnarok inspect` takes the absolute path of a note in such a vault.
- A note in a project whose `SUMMARY.md` has `state: done` or `state: abandoned` is `archived`. `stage: DONE`
  still archives a project. `find_projects` and `list_project` return the `state`.
- The `open_projects` tool lists the projects in progress and the paused projects, grouped by state. Its
  `states` argument selects other states, for example `abandoned`.
- A note in a project whose `SUMMARY.md` has `stage: DONE` is `archived`. An edit to a project marker file
  derives the metadata of the notes in its project again on the next update.
- `backfill propose` does not write a status that comes from the `SUMMARY.md` stage, so the status follows
  the stage when the project opens again.
- `open_note` gives the outline of a note of more than 4,000 words: each H1-H3 section with its heading path,
  level, words and first sentence. `outline` forces the outline or the text. `heading` also takes the end of a
  heading path, such as `Plan > Results`, to select one of two sections with the same heading.

## [1.0.0] - 2026-10-23

First public release.

[1.2.0]: https://github.com/freyr/ragnarok/releases/tag/v1.2.0
[1.1.0]: https://github.com/freyr/ragnarok/releases/tag/v1.1.0
[1.0.0]: https://github.com/freyr/ragnarok/releases/tag/v1.0.0
