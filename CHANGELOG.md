# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The repository is `freyr/ragnarok-bundle`. It holds the server and the Claude Code plugin marketplace
  `ragnarok`; add the marketplace with `/plugin marketplace add freyr/ragnarok-bundle`.
- Server release tags are `server-vX.Y.Z`. The plugins have their own tags, `plugin-vX.Y.Z`, and their own
  changelog, `plugins/CHANGELOG.md`.

## [2.1.0] - 2026-11-10

### Added

- The shared embedding cache `data/embed-cache.db`. Each index file attaches it. Its key is the model, the
  dimension, the document template and the text hash, so index files with the same embedding settings share the
  entries. `ragnarok index --rebuild` deletes only the index file and keeps the cache.
- A model profile for each shortlisted embedding model: the query prefix, the document template (with the title and
  the heading path), the native dimension and the token limit. The config keys `query_prefix` and `doc_template`
  override the profile.
- Variants and corpora: the tables `[variants.<name>]` (they override `model`, `dimension`, `query_prefix`,
  `doc_template`, `min_words`, `max_words`, `chunk_prefix` and `note_rank`) and `[corpora.<name>]` (`root` and
  `vaults`). Each pair has its own index file `data/variants/<variant>/<corpus>/index.db`. The MCP server always
  uses `data/index.db`.
- `chunk_prefix` (`none`, `description`) and `note_rank` (`maxp`, `topk_sum`, `note_vector`). The defaults are
  `none` and `maxp`, which is the search of 2.0.0.
- The bench commands: `ragnarok bench index` builds the index of a variant on a corpus, `ragnarok bench run` runs a
  question set of format v2 through the search tool, `ragnarok bench compare` gives recall@5 and MRR, the paired
  difference with a cluster bootstrap interval and the Holm correction, a verdict line for each variant, and the
  p95 latency. `ragnarok bench convert` converts a draft set to format v2. A set whose expected note is not in the
  index stops the run and lists each such path.

### Changed

- The index has schema version 3. The first start of 2.1.0 copies the embedding cache of the index file to the
  shared cache with no new embedding. The old cache table stays in the index file until the next
  `ragnarok index --rebuild`.
- `embed.py` sends `truncate: false`, and `dimensions` when the dimension is below the native one. The splitter
  cuts a chunk to the token limit of the profile. When Ollama refuses a text for its context length, the text is
  embedded in two halves and the vector is their mean. No text is cut silently.
- The config key `doc_prefix` is now `doc_template`. A `config.toml` with `doc_prefix` gives a warning and still
  works.

### Upgrade

- A 2.0.0 MCP server that still runs during the first 2.1.0 start keeps working, because the old cache table stays.
  A new start of 2.0.0 on a 2.1.0 index is refused.

## [2.0.0] - 2026-11-09

### Removed

- The tools `validate`, `find_collisions`, `vault_rules` and `find_entities`, and the commands
  `ragnarok validate`, `ragnarok find-collisions` and `ragnarok replay`. The knowledge-base model goes with them:
  the labels `class`, `validity`, `as_of`, `subject` and `superseded_by`, the entity notes, the `subject` filter
  and the `include_superseded` argument of `search`, and the label output of `search` and `list_project`.
- The `[write]` table of `config.toml` (`known_keys`, `systems_of_record`). A `config.toml` that still has the
  table gives a warning at the first load; ragnarok ignores the table. Remove it.

### Added

- The top-level key `keep_alive` (default `-1`) sets how long Ollama keeps the embedding model loaded: an
  integer of seconds, `-1` for always, or a duration such as `"30m"`.

### Changed

- The index has schema version 2. The first start of 2.0.0 migrates an index of version 1 in place and drops the
  tables and columns of the knowledge-base model, with no new embedding.

### Upgrade

- Stop every 1.x MCP server before the first 2.0.0 start: close each Claude Code session. A 1.2.0 server that
  starts after the migration adds the dropped tables back and asks for a rebuild of the index.

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

[2.0.0]: https://github.com/freyr/ragnarok-bundle/releases/tag/v2.0.0
[1.2.0]: https://github.com/freyr/ragnarok-bundle/releases/tag/v1.2.0
[1.1.0]: https://github.com/freyr/ragnarok-bundle/releases/tag/v1.1.0
[1.0.0]: https://github.com/freyr/ragnarok-bundle/releases/tag/v1.0.0
