Generating the docs
----------

Uses [MkDocs](https://www.mkdocs.org/) with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

Install doc dependencies:

    pip install -e ".[doc]"

Build and preview locally (from the repo root):

    # Build the site into ../site (exit 0 = success)
    python -m mkdocs build --config-file docs/mkdocs.yml

    # Serve with live-reload on http://127.0.0.1:8000
    python -m mkdocs serve --config-file docs/mkdocs.yml
