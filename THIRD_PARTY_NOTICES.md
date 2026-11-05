# Third-party sources and release status

The external OpenPI and LeRobot code is under Apache-2.0; LIBERO code is
under MIT. License texts from the checked local revisions are preserved in
`third_party/licenses/`. Source patches in this package modify these external
projects; retain their notices when applying or redistributing them.

- OpenPI: https://github.com/Physical-Intelligence/openpi
- LeRobot: https://github.com/huggingface/lerobot
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- HF LIBERO package: https://github.com/huggingface/LIBERO (`hf_libero==0.1.3`, metadata: MIT).

These source-code licenses do not establish a distribution license for the
trained checkpoints, patches, activation datasets, simulation textures/meshes,
or other upstream model assets. The artifact manifest records these licenses
as unconfirmed. In particular the local SmolVLA checkpoint configuration has
`license: null` and its cached snapshot has no model card/license file.
Do not infer its weight license from the LeRobot code license.

The license for this research package, selected
feature metadata, author changes in external patches, and author-trained
artifacts has not been supplied. No license has been chosen on the authors'
behalf. Resolve this before public release, along with any required notices
for simulator assets and base-model/tokenizer terms. This repository is in
release preparation and has not been pushed or published.
