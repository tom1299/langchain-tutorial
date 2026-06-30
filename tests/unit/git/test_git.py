import logging
import os
import sys
from typing import NamedTuple, Union, Optional
from unittest import skip

from pytest import fixture, mark

from lctutorial import init_chat_model

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
    prev_version: str
    prev_commit_sha: str

def get_diff_details(commit_sha: str, repo: Repo, file_path: str) -> Optional[DiffDetails]:
    commit = repo.commit(commit_sha)

    commits = list(repo.iter_commits(paths=file_path))

    # TODO: Refactor / improve code
    for i, c in enumerate(commits):
        if c.hexsha == commit_sha:
            if i + 1 < len(commits):
                parent_commit = commits[i + 1]
                previous_version = parent_commit.tree[file_path].data_stream.read().decode("utf-8")
                diffs = commit.diff(parent_commit, paths=os.path.join(repo.working_dir, file_path), create_patch=True)
                for diff in diffs:
                    new_version = commit.tree[file_path].data_stream.read().decode("utf-8")
                    return DiffDetails(diff=diff.diff.decode("utf-8"), commit_message=commit.message,
                                       new_version=new_version, prev_version=previous_version, prev_commit_sha=parent_commit.hexsha)

    return None

@mark.parametrize("path_to_git_repo", ["kubernetes_documentation"])
class TestGitApi:

    def test_get_diff_details(self, path_to_git_repo, request):
        repo = Repo(request.getfixturevalue(path_to_git_repo))
        service_md = "content/en/docs/concepts/services-networking/service.md"
        diff_details = get_diff_details("4a1f03802b423637c04c56258f87630a28139217", repo, service_md)

        assert diff_details.diff is not None
        assert diff_details.prev_commit_sha == '3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113'
        assert diff_details.commit_message is not None
        assert "Consider using an external load balancer controller or a Gateway API implementation instead" in diff_details.new_version
        assert "Consider using an external load balancer controller or a Gateway API implementation instead" not in diff_details.prev_version

@mark.parametrize("path_to_git_repo", ["kubernetes_documentation"])
@mark.parametrize(
    ("provider", "model"),
    [
        ("OpenAI", "gpt-4.1-mini"),
#        ("anthropic", "claude-3-5-sonnet"),
    ],
    ids=["gpt-4.1-mini"]#, "anthropic-sonnet"],
)
@skip("Skipping test_get_diff_details_with_model due to API key issues")
class TestAgentGitAnalysis:
    pass

    def test_get_diff_details_with_model(self, path_to_git_repo, request, provider, model):
        repo = Repo(request.getfixturevalue(path_to_git_repo))
        service_md = "content/en/docs/concepts/services-networking/service.md"
        diff_details = get_diff_details("3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113", repo, service_md)

        model = init_chat_model(provider=provider, model_name=model)

        prompt = f"""
        You are a helpful assistant that analyzes git diffs for documentation changes.
        You get the git diff and the version of the document before the change.
        Your task is to evaluate the changes made in a git commit.
        * Topic of change
        * Impact of the change to the overall content of the document 
        * Impact of the change to the readability of the document
        =========
        Git diff:
        {diff_details.diff}
        ========
        Previous version:
        {diff_details.prev_version}
        ========
        """
        response = model.invoke(prompt)
        print(response)
