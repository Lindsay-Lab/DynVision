# Evaluation Metrics

This reference describes the evaluation metrics DynVision computes during
training and testing, and how they generalize to temporally extended inputs. The
core implementations are in `dynvision/utils/performance_measures.py`.

## Description

DynVision's metrics are computed and aggregated across models and testing
experiments within the Snakemake workflow. All metrics respect the temporal
dimension of model outputs, so each metric is defined per timestep.

## Accuracy

Classification accuracy compares the predicted class index (`guess_index`) with
the true label index (`label_index`):

$$\text{Accuracy}(t) = \frac{1}{N}\sum_{n=1}^{N} \left[\,\text{guess\_index}_n(t) = \text{label\_index}_n(t)\,\right]$$

Timesteps whose label index is negative (the `non_label_index`, default `-1`)
are masked out and excluded from the average. If every label in a batch is
masked, the accuracy is reported as `0.0`.

- **During training** — computed for timesteps where the input has propagated
  through the network to the classifier (excluding residual timesteps and the
  `loss_reaction_time` offset).

- **During testing** — computed for *all* timesteps, enabling analysis of
  classification dynamics during both stimulus presentation and subsequent
  null-input periods.

### Top-k Accuracy

Top-$k$ accuracy checks whether the true label appears among the $k$ highest
scoring classes of the model output at each timestep. It is configured through
the `accuracy_topk` measure option and computed via `calculate_topk_accuracy`.

## Confidence

Confidence measures are derived from the softmax over the classifier output at
each timestep (`calculate_confidence`):

- **Guess confidence** (`guess_confidence`) — the maximum softmax probability,
  i.e. the certainty assigned to the predicted class:
  $\max_i \mathrm{softmax}(c)_i(t)$.

- **Label confidence** (`label_confidence`, `first_label_confidence`) — the
  softmax probability assigned to the *true* class at the given index:
  $\mathrm{softmax}(c)_{\text{label}}(t)$.

Indices that are out of range or negative yield a confidence of `0`.

## Average Response

Per-layer activations are captured via a `record` operation placed in the
`layer_operations` list. By default activations are recorded after each layer's
nonlinearity. Spatially averaged per-layer responses provide a first-order proxy
for comparison with electrophysiological data such as ECoG.

## See Also

- [Layer Operations](layer-operations.md) — where `record` fits in the pipeline
- [Benchmarking](benchmarking.md) — computational cost of metric tracking
- [Model Testing (How-to)](../user-guide/model-testing.md)
