# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-15

### Added

- Installable `online-gcs-planner` package with the `online_gcs` Python API and
  `online-gcs` command-line interface.
- Online GCS, RRT*, trajectory optimization, IRIS region construction, and
  parallel exploration components in a standard `src` layout.
- Reproducible examples, automated tests, package checks, and continuous
  integration.
- Matching English and Russian README files and documentation.

### Changed

- Moved supported scene configuration into package resources.
- Separated planning, metrics, visualization, scene, and CLI responsibilities.

### Removed

- Reports, notebooks, generated experiment results, logs, and other research
  artifacts from the current public tree.

[0.1.0]: https://github.com/qrvmil/GCS/releases/tag/v0.1.0
