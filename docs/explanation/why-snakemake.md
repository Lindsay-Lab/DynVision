# Why Snakemake?

!!! note "Understanding-oriented"
    This page explains the *reasoning* behind DynVision's use of Snakemake for
    workflow orchestration. For practical steps, see the
    [Workflow Management how-to guide](../user-guide/workflows.md).

## The problem: reproducible, large-scale experiments

DynVision experiments typically involve a combinatorial space of models,
recurrence types, datasets, and stimulus-presentation regimes. Running these by
hand is error-prone and hard to reproduce. A typical study sweeps over:

- Multiple model architectures (`DyRCNNx4`, `DyRCNNx8`, reference models)
- Several recurrence types (`full`, `self`, `local`, `depthpointwise`, …)
- Different datasets and data groups
- A range of temporal-presentation configurations

The number of resulting jobs grows multiplicatively, and many of them share
intermediate artifacts (prepared datasets, trained checkpoints).

## Why a workflow manager (and why Snakemake specifically)

Snakemake addresses three needs that are central to computational-neuroscience
research software:

| Need | How Snakemake addresses it |
|------|----------------------------|
| **Reproducibility** | Rules declare explicit inputs/outputs; re-running only recomputes what changed. |
| **Scalability** | The same workflow runs locally or on a SLURM cluster via executor plugins, with no code changes. |
| **Dependency tracking** | Intermediate artifacts (datasets, checkpoints, responses) are shared across jobs instead of recomputed. |

Snakemake's Python-based rule syntax also integrates naturally with DynVision's
configuration system, allowing parameter sweeps to be expressed as wildcards.

## Trade-offs and alternatives

Snakemake is not the only option, and it carries a learning curve. Alternatives
considered include plain shell scripts (no dependency tracking), Make (awkward
for Python and parameter sweeps), and heavier orchestrators such as Nextflow or
Airflow (more infrastructure than a research lab typically needs).

!!! tip "You don't have to use Snakemake"
    DynVision's model and training components are usable directly from Python.
    Snakemake is the *recommended* path for reproducible experiment sweeps, not
    a hard requirement. See the
    [Workflow Management guide](../user-guide/workflows.md) for the practical
    entry point.

## See also

- How-to: [Workflow Management](../user-guide/workflows.md)
- How-to: [Cluster Integration](../user-guide/cluster-integration.md)
- Reference: [Dependency notes — Snakemake](../development/dependencies/snakemake.md)
