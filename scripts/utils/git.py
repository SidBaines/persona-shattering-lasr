"""Git utilities for enforcing script provenance."""

from __future__ import annotations

from pathlib import Path

import git


def assert_clean_and_pushed(allow_override: bool = True) -> str:
    """Assert that the repo has no uncommitted changes and HEAD is pushed.

    Checks:
      1. No uncommitted changes exist anywhere in the repo (staged or unstaged)
         relative to the last commit.
      2. The current HEAD commit exists on the tracked remote branch.

    If either condition fails, the changed files are reported. When
    ``allow_override`` is True, the user is prompted to continue anyway;
    otherwise a RuntimeError is raised immediately.

    Args:
        allow_override: If True, prompt the user to override on failure instead
            of raising immediately.

    Returns:
        The current HEAD commit SHA.

    Raises:
        RuntimeError: If conditions are not satisfied and the user does not
            override (or ``allow_override`` is False).
    """
    try:
        repo = git.Repo(Path.cwd(), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise RuntimeError("No git repository found from the current directory.")

    errors: list[str] = []

    # --- Check 1: uncommitted changes (staged or unstaged) ---
    staged = repo.index.diff("HEAD")
    unstaged = repo.index.diff(None)
    untracked = repo.untracked_files

    dirty_files: list[str] = []
    for diff in staged:
        dirty_files.append(f"  staged:    {diff.a_path}")
    for diff in unstaged:
        dirty_files.append(f"  modified:  {diff.a_path}")
    for path in untracked:
        dirty_files.append(f"  untracked: {path}")

    if dirty_files:
        errors.append(
            "Repo has uncommitted changes:\n" + "\n".join(dirty_files)
        )

    # --- Check 2: HEAD has been pushed ---
    commit_sha = repo.head.commit.hexsha
    tracking_branch = repo.active_branch.tracking_branch()

    if tracking_branch is None:
        errors.append(
            f"Branch '{repo.active_branch.name}' has no upstream tracking branch.\n"
            "  Push it first: git push -u origin HEAD"
        )
    else:
        remote_name = tracking_branch.remote_name
        repo.remotes[remote_name].fetch()
        remote_commit = tracking_branch.commit

        if not repo.is_ancestor(commit_sha, remote_commit.hexsha):
            errors.append(
                f"HEAD commit {commit_sha[:12]} has not been pushed to "
                f"'{tracking_branch.name}'.\n"
                "  Push first: git push"
            )

    if not errors:
        return commit_sha

    # --- Report and optionally prompt ---
    print("\n" + "=" * 60)
    print("WARNING: Git provenance checks failed")
    print("=" * 60)
    for error in errors:
        print(error)
    print("=" * 60)

    if allow_override:
        response = input(
            "\nContinue anyway? The commit hash logged to W&B/HF may not "
            "reflect the exact code that was run. [y/N] "
        ).strip().lower()
        if response == "y":
            print(f"Continuing with commit {commit_sha[:12]} (unverified).\n")
            return commit_sha

    raise RuntimeError(
        "Aborting: commit and push all changes before running, or confirm the override prompt."
    )
