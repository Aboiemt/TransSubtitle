# Publication plan

Target: `Aboiemt/TransSubtitle`, public repository, release `v0.1.0`.
The user requested an English introduction and a downloadable release, and then
explicitly removed the code-signing requirement. The Windows release is unsigned.

1. Inspect the application, dependencies, local configuration and build inputs.
2. Back up publication-related files outside the repository before editing them.
3. Write an English README and add a clean release builder. Keep application
   behavior unchanged. Never copy `.env` or private `settings.json` into a release.
4. Build with the project virtual environment and include the local Whisper base
   model. Include its upstream MIT license and dependency license notices.
5. Check imports, dependency consistency, source CLI, audio device enumeration,
   local CPU inference, and the frozen application's launch. Record actual results.
6. Inspect the Git staging area for credentials, generated files and large models.
7. Create the repository, push the source and version tag, upload the Windows ZIP,
   checksum and verification report, and publish the release.
8. Verify the remote source, tag, release assets and uploaded SHA-256 digest.

Backups: `G:\Program Files\codexcache\trans-publish\backup-20261106`.
Build/download caches: `G:\Program Files\codexcache`.

Publication status and measured checks are recorded in `docs/RELEASE_VALIDATION.md`.
