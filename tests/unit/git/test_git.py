import logging
import os
import sys

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
                # changed_document = commit.tree["content/en/docs/concepts/services-networking/service.md"]
                # print(changed_document.data_stream.read().decode("utf-8"))
                # previous_document = prev_commit.tree["content/en/docs/concepts/services-networking/service.md"]
                # print(previous_document.data_stream.read().decode("utf-8"))

            prev_commit = commit
