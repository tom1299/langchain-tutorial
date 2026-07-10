import logging
import os
import sqlite3
import sys
from select import select
from typing import NamedTuple, Union, Optional

from sqlite3 import Connection

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

@fixture(scope="class")
def database_connection() -> Connection:
    conn = sqlite3.connect("file::memory:?cache=shared", uri=True)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS change_description (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_significance INTEGER,
            overall_significance INTEGER,
            main_topic TEXT,
            short_description TEXT,
            commit_sha TEXT,
            prev_commit_sha TEXT,
            heading TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            change_description_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            PRIMARY KEY (change_description_id, label),
            FOREIGN KEY (change_description_id) REFERENCES change_description(id)
        )
    """)
    conn.commit()
    return conn

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
    labels: list[str] = Field(..., description="A list of labels describing the topic of the change.")

class ChangeDescriptionList(BaseModel):
    changes: list[ChangeDescription] = Field(..., description="A list of changes made to the document. Derived from the git diff. Related parts of the git diff (same topic) are grouped together.")

# TODO: Refactor / use real ORM
def persist_change_description(conn: Connection, change: ChangeDescription) -> int:
    cursor = conn.execute(
        """
        INSERT INTO change_description
            (topic_significance, overall_significance, main_topic, short_description, commit_sha, prev_commit_sha, heading)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (change.topic_significance, change.overall_significance, change.main_topic,
         change.short_description, change.commit_sha, change.prev_commit_sha, change.heading),
    )
    change_id = cursor.lastrowid
    conn.executemany(
        "INSERT INTO labels (change_description_id, label) VALUES (?, ?)",
        [(change_id, label) for label in change.labels],
    )
    conn.commit()
    return change_id

# TODO: Refactor / use real ORM
def load_change_descriptions(conn: Connection) -> list[ChangeDescription]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT cd.*, l.label
        FROM change_description cd
        LEFT JOIN labels l ON l.change_description_id = cd.id
        ORDER BY cd.id
    """).fetchall()

    changes: dict[int, dict] = {}
    for row in rows:
        cd_id = row["id"]
        if cd_id not in changes:
            changes[cd_id] = {**dict(row), "labels": []}
        if row["label"]:
            changes[cd_id]["labels"].append(row["label"])

    return [ChangeDescription(**{k: v for k, v in cd.items() if k != "label"})
            for cd in changes.values()]

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

def get_diff_details_with_model(path_to_git_repo, md:str, model) -> ChangeDescriptionList:
    repo = Repo(path_to_git_repo)
    diff_details = get_diff_details("3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113", repo, md)

    labels = ["kubernetes", "services", "endpoint-slices", "service-discovery", "load-balancing"]
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
    * For the labels field, use at least one of the following labels: {", ".join(labels)} which is appropriate for the change.
    * Also select or use one of the existing labels to describe the change content. Avoid generic labels like "documentation-update". Use labels that express what was semantically done. For example if the change updated a RFC reference an appropriate label would be "RFC-12345-update"
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

    # TODO: Overall significance and topic significance seem to be still very high.
    response = model.invoke(prompt)
    return response["parsed"]

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
        repo_path=request.getfixturevalue(path_to_git_repo)
        service_md = "content/en/docs/concepts/services-networking/service.md"
        model = init_chat_model(provider=provider, model_name=model)
        model_with_structure = model.with_structured_output(ChangeDescriptionList, include_raw=True)



@mark.parametrize("path_to_git_repo", ["kubernetes_documentation"])
@mark.usefixtures("database_connection")
@mark.parametrize(
    ("provider", "model"),
    [
        ("OpenAI", "gpt-4o-mini"),
#        ("anthropic", "claude-3-5-sonnet"),
    ],
    ids=["gpt-4o-mini"]#, "anthropic-sonnet"],
)
class TestGitHistoryAnalysis:

    def test_analyze_git_history(self, path_to_git_repo, database_connection: Connection, request, provider, model):
        repo_path=request.getfixturevalue(path_to_git_repo)
        service_md = "content/en/docs/concepts/services-networking/service.md"
        model = init_chat_model(provider=provider, model_name=model)
        model_with_structure = model.with_structured_output(ChangeDescriptionList, include_raw=True)

        response = get_diff_details_with_model(repo_path, service_md, model_with_structure)
        assert len(response.changes) == 2

        for change in response.changes:
            persist_change_description(database_connection, change)

        changes = load_change_descriptions(database_connection)
        assert len(changes) == 2
        for change in changes:
            print(f"Change: {change.short_description}, labels: {change.labels}, topic significance: {change.topic_significance}, overall significance: {change.overall_significance}")


