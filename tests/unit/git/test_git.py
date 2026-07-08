import logging
import os
import sys
from typing import NamedTuple, Union, Optional
from unittest import skip

from pydantic import BaseModel, Field
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
    commit_sha: str
    prev_commit_sha: str

class ChangeDescription(BaseModel):
    """Details about the changes made to a document."""
    topic_significance: int = Field(0, ge=0, le=100, description="The significance of the change to the semantic / meaning of the topic it covers. Between 0 and 100. Topic should be the main topic. See field 'main_topic' for more details.")
    overall_significance: int = Field(0, ge=0, le=100, description="The overall significance of the change to the semantic / meaning to the complete document. Between 0 and 100.")
    main_topic: str = Field(..., description="The main topic in the document this change is related to.")
    short_description: str = Field(..., max_length=200, description="A short description of the change made to the document.")
    commit_sha: str = Field(..., description="The commit SHA of the change.")
    prev_commit_sha: str = Field(..., description="The commit SHA of the previous commit.")
    heading: str = Field(..., description="The heading in the document under which the change was made.")

class ChangeDescriptionList(BaseModel):
    changes: list[ChangeDescription] = Field(..., description="A list of changes made to the document. Derived from the git diff. Related parts of the git diff (same topic) are grouped together.")

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
                                       new_version=new_version, prev_version=previous_version, prev_commit_sha=parent_commit.hexsha,
                                       commit_sha=commit.hexsha)

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
        ("OpenAI", "gpt-4o-mini"),
#        ("anthropic", "claude-3-5-sonnet"),
    ],
    ids=["gpt-4o-mini"]#, "anthropic-sonnet"],
)

class TestAgentGitAnalysis:
    pass

    def test_get_diff_details_with_model(self, path_to_git_repo, request, provider, model):
        repo = Repo(request.getfixturevalue(path_to_git_repo))
        service_md = "content/en/docs/concepts/services-networking/service.md"
        diff_details = get_diff_details("3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113", repo, service_md)

        # TODO: Refactor this prompt
        # - Add an example
        prompt = f"""
        You are a helpful assistant that analyzes git diffs for documentation changes.
        You get the git diff and the version of the document before the change.
        Use the diff and the previous document version to analyze the changes made.
        Your task is to evaluate the changes made in a git commit and return 
        an instance of the class ChangeDescriptionList for each change contained in the diff.
        Use the git diff to do the following:
        * For each change in the diff create a ChangeDescription instance. See the fields of the class ChangeDescriptionList for details what to put where.
        * If changes are semantically closely related, merge them into a single ChangeDescription instance.
        * Return the list of ChangeDescription instances.
        =========
        Git diff:
        {diff_details.diff}
        ========
        Previous document version:
        {diff_details.prev_version}
        ========
        Commit sha:
        {diff_details.commit_sha}
        ========
        Previous commit sha:
        {diff_details.prev_commit_sha}
        """
        model = init_chat_model(provider=provider, model_name=model)
        model_with_structure = model.with_structured_output(ChangeDescriptionList, include_raw=True)

        # TODO: Overall significance and topic significance seem to be still very high.
        response = model_with_structure.invoke(prompt)
        assert len(response['parsed'].changes) == 2
