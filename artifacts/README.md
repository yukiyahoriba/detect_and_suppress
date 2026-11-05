# Artifact inventory and retrieval

`manifest.json` identifies the current representative experiments by model,
role, portable destination, byte size, SHA256, required use, file set (for
directories), known upstream identity, distribution plan and license status.
No weights, patch images, activation collections, or caches are embedded in
the source package. Feature statistics are in `selections/`; their metadata
bind the original SAE by checksum, so relocation needs no private paths.
The statistics, feature IDs and ranking are unchanged; only provenance paths
were replaced by content identities. Original selection checksums remain in
the new metadata. These are the *recorded chosen features*, not a claim that
a fresh top-1 search would choose them from every available layer.

| Model | VLA checkpoint | SAE | Probe | Patch | Additional runtime assets |
|---|---:|---:|---:|---:|---:|
| Pi0.5 | 44,745,151,753 B | 33,627,378 B | 26,616 B | 8,194 B | 4,264,023 B tokenizer |
| SmolVLA | 1,218,052,342 B | 7,442,046 B | 13,512 B | 7,552 B | 2,034,845,165 B VLM/processor/tokenizer |

The source manifest contains exact sizes/checksums for the small selection
JSON files too. Large author artifacts should be hosted outside GitHub;
SAE and Probe weights should use the same versioned external release for
consistent identity, even though some files are small. Selection JSON,
checksums and experiment templates belong in source control after rights
are confirmed. Patch redistribution requires confirmation even though PNGs
are small. The proposed hosting service/repository has **not been chosen**.
Manifest `url: null` means there is no approved download, not an omitted
working URL. Do not replace missing paper weights with another checkpoint.

Known upstream identities:

- SmolVLA: `HuggingFaceVLA/smolvla_libero`, commit
  `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- SmolVLA runtime: `HuggingFaceTB/SmolVLM2-500M-Instruct`, commit
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Pi0.5 tokenizer: `gs://big_vision/paligemma_tokenizer.model`; use the
  manifest's content hash because this URL does not identify an immutable version.

The HF identities were confirmed from local snapshots and cache refs. A
fresh network download and current upstream model-card/license could not
be verified here. Use upstream tooling to obtain these revisions under
its terms, restrict files to the manifest's `files`, and verify checksums.
Extra README/cache metadata files change a directory checksum. A directory
hash is SHA256 of each sorted relative POSIX filename + NUL + file contents,
concatenated in order; symlink targets are read. It is not the hash of a
compressed archive or of concatenated per-file digests.

Once approved artifacts are available, arrange:

```
weights/
  pi05/{checkpoint/,sae.npz,probe.npz,patch.png,selection.json,runtime.model}
  smolvla/{checkpoint/,sae.npz,probe.npz,patch.png,selection.json,runtime/}
```

Copy `selections/pi05.json` to `weights/pi05/selection.json`, and likewise
for SmolVLA. Keep any legacy Probe JSON sidecar beside its NPZ when supplied;
legacy NPZ without full metadata is explicitly bound by the evaluation
configuration, as in the compatibility-tested API. It is not recalibrated.

Run `das verify-artifacts --manifest artifacts/manifest.json --artifact-root weights`
(or `--model pi05`). Verification never downloads or substitutes assets.
Then run `das prepare-runtime` ([execution guide](../docs/running_experiments.md)), which copies only verified auxiliary
files into an isolated native cache. SmolVLA's HF `refs/main` is explicitly
bound to the inventoried revision; offline mode prevents a silent update.
Native inference code and original weight-loading choices are unchanged.

Full Feature Selection/Probe retraining also needs the activation collections,
episode labels and splits. These are not required for evaluation with saved
artifacts. Their release location, size inventory and distribution rights
are still pending; the synthetic example is deliberately separate.
