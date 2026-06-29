## Important uv commands
Update all dependencies:
```bash
uv lock --upgrade
uv sync
```
Use dry run to see changes:
```bash
uv lock --upgrade --dry-run
```
Export all dependencies to requirements.txt format:
```bash
uv export --format requirements-txt > requirements.txt
```
Install playwright deps after upgrade:
```aiignore
uv run playwright install --with-deps
```
Run a specific test with a fixture:
```
uv run pytest tests/unit/git/test_git.py::TestGitApi::test_get_diff_for_file_and_commits[kubernetes_documentation]
```