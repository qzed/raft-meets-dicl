from pathlib import Path

import git


def get_git_head_hash(default=None, pfx_dirty='~'):
    try:
        repo = git.Repo(Path(__file__).parent, search_parent_directories=True)

        if repo.is_dirty():
            return pfx_dirty + repo.head.object.hexsha
        else:
            return repo.head.object.hexsha

    except git.exc.InvalidGitRepositoryError:
        return default
