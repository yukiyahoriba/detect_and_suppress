# SAE feature selection

Feature selection uses saved residual activations and patch-presence signals;
it does not run VLA inference, train an SAE or depend on Probe training.

```python
from detect_and_suppress.sae_checkpoint import load_sae_encoder, encode_activations
from detect_and_suppress.selection import (
    EpisodeLatents, compute_correlations, aggregate_correlations,
    rank_features, select_features, save_selection, load_selection,
)

sae = load_sae_encoder("sae.npz")
episode = EpisodeLatents(
    expert="pg", layer=5, episode_id=0, task_id=0,
    latents=encode_activations(activations, sae),
    patch_signal=patch_presence,
    policy_forward_signal=policy_forwards,
)
rows = compute_correlations([episode])
ranking = rank_features(aggregate_correlations(rows, threshold=0.6), criterion="mean")
selection = select_features(ranking, top_n=20, metadata={"source": "saved-episodes"})
save_selection("selection.json", selection)
selection = load_selection("selection.json")
```

Activations have shape `[samples, hidden_dim]`. Encoding uses the checkpoint's
training TopK, defaulting to 100 when absent; `load_sae_encoder` accepts an
explicit override. It preserves the research NumPy encoding path.

`EpisodeLatents` contains `[steps, features]` latents and one-dimensional patch
and policy-forward signals. Signals are truncated to their common length.
Only samples with policy-forward signal greater than 0.5 contribute. Features
that never fire are absent from the correlation table; undefined correlations
are `None` and excluded from episode statistics. Aggregate means are unweighted
across valid episodes, independent of episode length.

`rank_features` supports mean correlation, maximum correlation and episode
count above the supplied threshold. These rankings include nonpositive and
undefined scores. The separate `mean_feature_scores` / `select_positive_features`
API reproduces positive-only selection. Do not interchange these rules when
reproducing a recorded experiment.

`read_correlations` / `write_correlations` handle the existing episode Pearson
CSV layout. Selection JSON retains feature identities, summary statistics,
ranking rule, threshold and metadata. Bind a selection to its SAE using
`metadata["sae_sha256"]` when relocating checkpoints; otherwise the existing
path-binding rules apply. The CLI records this hash automatically.

CLI input format: [configurations](../configs/README.md).
