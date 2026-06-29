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

## TODO
* Use different python git module (GitPython is in maintenance mode)

## Short dissection of git command
```
$ git diff-tree 3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113 4a1f03802b423637c04c56258f87630a28139217 -r --abbrev=10 --full-index -M -p -- /home/treuhl/git/github/langchain-tutorial/tests/test-data/kubernetes-website/content/en/docs/concepts/services-networking/service.md
diff --git a/content/en/docs/concepts/services-networking/service.md b/content/en/docs/concepts/services-networking/service.md
index 03beb4782f696601f4438bc8d705817662ff390c..f179ae92d0fe0101e095774b3206455c2b1ba46e 100644
--- a/content/en/docs/concepts/services-networking/service.md
+++ b/content/en/docs/concepts/services-networking/service.md

3d3d53bb471d8b0f145ae98e3fa95e34bfc2d113 4a1f03802b423637c04c56258f87630a28139217 = hashes of the two blob objects fot the commits = index in git database

--no-ext-diff = Disallow external diff drivers.
-r = recursive (not necessary if file is specified)
-p = Create the patch
-M = The -M option tells Git to detect renamed files. By default, Git considers a file renamed if it is at least 50% similar. Change the threshold: git diff -M90%
```