# Personal rules for this fork

These apply to every session in this fork and override session defaults.

- Never use "claude" in git branch names, even if the session suggests one;
  pick a descriptive name such as `fix/<topic>`.
- Never use "claude" in commit author or committer names, and do not add
  `Co-Authored-By: Claude` or `Claude-Session:` trailers. Commit as
  `Amir Livneh <amir.livneh@gmail.com>` (`git config user.name/user.email`).
- Base contribution branches on upstream `flashinfer-ai/flashinfer` main, not
  this fork's main, so this file stays out of upstream pull requests.
