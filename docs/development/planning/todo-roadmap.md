## Key Documentation Improvements (2026-06-30 / 2026-07-01)

The following high-level documentation tasks have been completed during the docs‑website work and follow-up todo sweep:

### Docs‑Website Overhaul (2026-06-30)
- ✅ **Diátaxis nav restructure** — index pages use frontmatter titles, quadrant badges via overrides
- ✅ **Link audit** — 40+ broken links fixed (tutorials→tutorial, evaluation→model-testing, docs/ prefix)
- ✅ **Image handling** — manuscript figures copied to docs/assets, paths normalized (`../../assets/` prefix), non‑manuscript PNGs removed
- ✅ **Checklist rendering** — `pymdownx.tasklist` extension added
- ✅ **Flat-list fixes** — all files with bullet-list-to-paragraph issues corrected
- ✅ **Site‑name/header** — `site_name: ""` (logo serves as header), home page `title` frontmatter removed
- ✅ **Cluster guide** — `docs/user-guide/cluster-integration.md` fleshed out
- ✅ **New explanation pages** — `engineering-vs-biological-time.md`, `comparison-to-neural-data.md`
- ✅ **New reference pages** — `layer-operations.md`, `skip-feedback-connections.md`, `integration-strategies.md`, `evaluation-metrics.md`, `benchmarking.md`
- ✅ **Code‑of‑conduct** — filename corrected, added to not_in_nav
- ✅ **README.md** — fixed broken quick‑start params, docs badge, citation

### Todo Sweep (2026-07-01)
- ✅ **training.md** — new how‑to guide (verified against `train_model.py` + `snake_runtime.smk`)
- ✅ **5 reference pages fleshed out** — all verified against codebase/manuscript
- ✅ **Code‑vs‑doc audit** — all 4 flagged naming mismatches (#3, #14, #15, #16) verified/resolved
- ✅ **README** — `recurrency_types.png` figure added
- ✅ **UX** — commented aspirational links in index pages converted to visible roadmap notes
- ✅ **Planning files** — todo-docs.md, todo-release-0.1.md, todo-roadmap.md updated with current status