from pathlib import Path

import git


def get_git_head_hash(default=None):
    try:
        repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
        return repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        return default
