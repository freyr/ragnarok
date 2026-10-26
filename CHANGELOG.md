# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A note in a project whose `SUMMARY.md` has `stage: DONE` is `archived`. An edit to a project marker file
  derives the metadata of the notes in its project again on the next update.
- `backfill propose` does not write a status that comes from the `SUMMARY.md` stage, so the status follows
  the stage when the project opens again.

## [1.0.0] - 2026-10-23

First public release.

[1.0.0]: https://github.com/freyr/ragnarok/releases/tag/v1.0.0
