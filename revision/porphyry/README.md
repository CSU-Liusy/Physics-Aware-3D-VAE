# Optional external synthetic porphyry supplement

This folder contains reviewed source, a conversion recipe, small CPU fixture
tests and minimal source-bound release attestations. The registry records a
completed author-checkpoint fixed-ten production parity check and saved-only
recomputation of 150 author probability records. It enables the normal public
entry for a separately completed primary run; all primary-run and external-data
guards still apply. No external CSV, repository license, prepared voxel array,
prediction, weight, author cache or private orebody file is distributed here.

Obtain the upstream synthetic `datasets/porphyry_01/datafiles/block_model.csv`
and its repository license from commit `46e5f9cab80c6d69a412ceb4cd07ecc46bbed423`
of `exepulveda/geomet_datasets`. The source CSV must be 23,974,796 bytes with
SHA256 `deb4604875deb534b45463dbcbb81aced97c5453f9ae807077c7f9a802abbf38`.
The repository labels the data CC BY-NC-SA 4.0; retain attribution and comply
with its separate license. Do not add those inputs or derived arrays to this
code package. The paper citation is Garrido et al. (2020), Natural Resources
Research 29, 3527-3545, DOI 10.1007/s11053-020-09692-6.

From the public code root, use explicitly separate local paths:
Replace `ABSOLUTE_SOURCE_DIR` and `ABSOLUTE_PREPARED_DIR` with different absolute
directories outside this repository and any submission document tree.

```text
python porphyry/porphyry_public.py prepare --source ABSOLUTE_SOURCE_DIR/block_model.csv --license-file ABSOLUTE_SOURCE_DIR/LICENSE --code-root . --out ABSOLUTE_PREPARED_DIR --accept-data-license
python porphyry/porphyry_public.py inspect --primary-run runs/esi_revision_v2 --code-root . --data-cache ABSOLUTE_PREPARED_DIR
python -m unittest discover -s porphyry -p 'test_*.py' -v
```

The normal run, after separate numerical validation and source-bound release,
requires an independently trained, formal public primary run and retains all
five Cu levels at 0.1/0.2/0.3/0.5/1.0 percent and masks 42/53/67. It is 15
dependent conditions times ten methods = 150 newly generated predictions, all
from one external synthetic field. It is not field validation or a replacement
for the 750-case primary test. Inspection alone does not run models.

The separate author checkpoint route imports the original author's completed
pure-synthetic run explicitly, never relabels it as public training, and limits
the CUDA integration check to Cu 0.3, mask 42, seed 11 and ten methods. The
`verify-reference` command recomputes metrics from 150 saved author probability
records without new predictions. Author source artifacts and its original
budget must be provided separately; they are not bundled. The author fixed-ten
execution requires its separate internal supervisor and is not a standalone
public command in this folder. A successful fixed
ten-method software check or saved-reference check would not establish public
from-scratch training or a complete independent public 150-row evaluation.

The release evidence records ten actual author-checkpoint predictions with
bitwise-equal raw and projected outputs against the fixed internal condition,
plus 150 saved-only reference scores and independent source/scope review.
The scientific reference still has one synthetic parent and 15 dependent
conditions. Public independent full-150 prediction and public from-scratch
training have not been executed. Fixture tests remain software checks and do
not enlarge those scientific denominators. The attestations contain hashes and
bounded scopes; author paths, geometry, caches and weights remain excluded.
