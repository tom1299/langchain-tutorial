import logging
import os
import sys
from typing import NamedTuple, Union, Optional

from pytest import fixture, mark

from git import Repo

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@fixture(scope="module")
def kubernetes_documentation() -> str:
    test_file_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_file_path, "../../test-data/kubernetes-website")

class DiffDetails(NamedTuple):
    diff: str
    commit_message: Union[str, bytes, None]
    new_version: str

def get_diff_details(commit_hash: str, repo: Repo, file: str) -> Optional[DiffDetails]:
    commit = repo.commit(commit_hash)
    parent_commit = commit.parents[0] if commit.parents else None

    if parent_commit:
        diffs = commit.diff(parent_commit, paths=os.path.join(repo.working_dir, file), create_patch=True)
        for diff in diffs:
            new_version = commit.tree[file].data_stream.read().decode("utf-8")
            return DiffDetails(diff=diff.diff.decode("utf-8"), commit_message=commit.message, new_version=new_version)

    return None

@mark.parametrize("path_to_git_repo", ["kubernetes_documentation"])
class TestGitApi:

    def test_get_diff_for_file_and_commits(self, path_to_git_repo, request):
        logging.getLogger("test").info("Start test")
        repo = Repo(request.getfixturevalue(path_to_git_repo))
        service_md = os.path.join(repo.working_dir, "content/en/docs/concepts/services-networking/service.md")
        commits = repo.iter_commits(all=True, max_count=10, paths=service_md)
        prev_commit = None
        for commit in commits:
            if prev_commit:
                diffs = commit.diff(other=prev_commit, paths=service_md, create_patch=True)
                for diff in diffs:
                    print(commit.message, diff.diff)
                new_version = commit.tree["content/en/docs/concepts/services-networking/service.md"]

                # previous_document = prev_commit.tree["content/en/docs/concepts/services-networking/service.md"]
                # print(previous_document.data_stream.read().decode("utf-8"))

            prev_commit = commit

    def test_get_diff_details(self, path_to_git_repo, request):
        repo = Repo(request.getfixturevalue(path_to_git_repo))
        service_md = "content/en/docs/concepts/services-networking/service.md"
        diff_details = get_diff_details("4a1f03802b423637c04c56258f87630a28139217", repo, service_md)

        assert diff_details.diff is not None
        assert diff_details.commit_message is not None
        assert "Consider using an external load balancer controller or a Gateway API implementation instead" in diff_details.new_version
