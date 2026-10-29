# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[1.0.0]: https://github.com/freyr/ragnarok/releases/tag/v1.0.0
