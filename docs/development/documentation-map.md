# Documentation Map (Diátaxis)

This page documents how DynVision's documentation is organised according to the
[Diátaxis framework](https://diataxis.fr/). It serves as the content-mapping
reference for maintainers.

## The four quadrants

| Quadrant | Orientation | Reader is… | Accent |
|----------|-------------|------------|:------:|
| **Tutorials** | Learning | …studying; following a guided path | 🟢 Green `#2ecc71` |
| **How-to Guides** | Tasks | …working; solving a specific problem | 🟫 Brown `#795548` (brand accent `#CF8A2A`) |
| **Reference** | Information | …looking something up | ⬛ Slate `#34495e` |
| **Explanation** | Understanding | …reading to understand *why* | 🟦 Teal `#16a085` |

## Content mapping

The on-disk directory names are intentionally **kept unchanged** to preserve
existing URLs. The Diátaxis grouping is expressed in the navigation and styling
layer. The table below records the mapping.

| Current page (on disk) | Diátaxis quadrant | Nav label |
|------------------------|-------------------|-----------|
| `getting-started.md` | Tutorials | Getting Started |
| `tutorial/basic-model-training.md` | Tutorials | Basic Model Training |
| `tutorial/custom-model.md` | Tutorials | Custom Model Creation |
| `user-guide/installation.md` | How-to Guides | Installation |
| `user-guide/custom-models.md` | How-to Guides | Custom Models |
| `user-guide/data-processing.md` | How-to Guides | Data Processing |
| `user-guide/temporal-data-presentation.md` | How-to Guides | Temporal Data Presentation |
| `user-guide/workflows.md` | How-to Guides | Workflow Management |
| `user-guide/model-testing.md` | How-to Guides | Model Testing |
| `user-guide/visualization.md` | How-to Guides | Visualization |
| `user-guide/parameter-handling.md` | How-to Guides | Parameter Handling |
| `user-guide/cluster-integration.md` | How-to Guides | Cluster Integration |
| `user-guide/troubleshooting.md` | How-to Guides | Troubleshooting |
| `reference/organization.md` | Reference | Organization |
| `reference/model-components.md` | Reference | Model Components API |
| `reference/recurrence-types.md` | Reference | Recurrence Types |
| `reference/dynamics-solvers.md` | Reference | Dynamics Solvers |
| `reference/configuration.md` | Reference | Configuration |
| `reference/losses.md` | Reference | Losses |
| `reference/optimizers-schedulers.md` | Reference | Optimizers & Schedulers |
| `explanation/biological-plausibility.md` | Explanation | Biological Plausibility |
| `explanation/temporal_dynamics.md` | Explanation | Temporal Dynamics |
| `explanation/design-philosophy.md` | Explanation | Design Philosophy |
| `explanation/why-snakemake.md` | Explanation | Why Snakemake? |
| `explanation/role-of-recurrence.md` | Explanation | Role of Recurrence |
| `development/**` | About (outside the 4 quadrants) | Developer Guide |
| `contributing.md` | About + footer | Contributing |
| `LICENSE` | Footer | — |

## Authoring rules per quadrant

To keep the quadrants distinct, follow these conventions when writing new pages:

=== "Tutorials"

    - Single, linear, immersive path; assume no prior knowledge.
    - Every step must produce a visible, working result.
    - No alternatives or digressions — that belongs in How-to or Explanation.
    - Use Previous/Next navigation.

=== "How-to Guides"

    - Start with a **Goal** callout stating what the reader will achieve.
    - List prerequisites up front.
    - Numbered, action-oriented steps. Assume working knowledge.
    - Link to Reference for parameter details rather than inlining them.

=== "Reference"

    - Dense, factual, consistent structure. No conversational filler.
    - Auto-generate API tables from docstrings via `mkdocstrings`.
    - Syntax-highlighted, copy-enabled code blocks.

=== "Explanation"

    - Discuss *why*, trade-offs, background, and alternatives.
    - Diagrams and conceptual sidebars encouraged.
    - Cross-link to the relevant How-to and Reference pages.

## Placeholder / planned pages

The Explanation index historically listed many aspirational pages that do not
yet exist (e.g. *Feedback Mechanisms*, *Time Constants*, *Comparison with
SNNs*). These are tracked in
[Documentation TODOs](planning/todo-docs.md) and should be created on demand
rather than left as broken links.
