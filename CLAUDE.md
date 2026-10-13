# Notes for sessions working on this repository

## Parag's dissertation (recorded 13 Oct 2026, asked once as the takeover prompt requires)

Cambridge masters dissertation, supervised by Dr Fatemeh Torabi (hence the `TORABI-*` CSD3 accounts).
Topic: **Bridge**, a retrospective diagnostic-accuracy study of a large language model that extracts
facts from Emergency Department fracture referral letters, combined with a deterministic rules engine
encoding the department's written virtual fracture clinic protocol. Primary outcome is the missed-review
rate against the consultant's recorded decision on 1,000 past referrals. The protocol is a separate
document; it is not in this repository and should stay out of it.

The Semataxy programme in this repository is a separate research track from the dissertation.

## Cluster facts (CSD3, verified 13 Oct 2026)

- Account `TORABI-SL3-CPU`, icelake partition, CPU only so far. Working tree at `~/rds/hpc-work/projects/frcs-678`.
- Batch steps see one core unless the body runs under `srun`; the `cambridge/*.sbatch` files do this.
- The cluster cannot reach GitHub with a key. Code is pushed from the laptop with `rsync` and commits are
  made on the laptop clone; results are pulled back the same way. See `docs/updates/` for the running log.
- Fixes made on first use are recorded in commit messages, as the takeover prompt asks.
