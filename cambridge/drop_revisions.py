"""Remove downloaded revisions of one model from the CSD3 Hugging Face cache once they are measured (25 Oct;
pg: "delete each after it's measured"). Dry run unless --delete.

A revision's snapshot goes only if no ref outside the list points at the same commit (Pythia's `main` is
the commit of step143000). A blob goes only if no remaining snapshot of the model links to it (config.json
is often shared with `main`); a blob that is itself a link into the shared store (hub/blobs/<xx>/<hash>,
see cambridge/free_cache.py) loses only that link, never the stored file.

    python cambridge/drop_revisions.py EleutherAI/pythia-410m --revs step0,step64,... [--delete]
"""
import argparse, os, shutil

from huggingface_hub import constants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("--revs", required=True); ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    root = os.path.join(constants.HF_HUB_CACHE, "models--" + a.repo.replace("/", "--"))
    refs = {r: open(os.path.join(root, "refs", r)).read().strip() for r in os.listdir(os.path.join(root, "refs"))}
    drop = [r for r in a.revs.split(",") if r in refs]
    keep_commits = {c for r, c in refs.items() if r not in drop}
    drop_commits = {refs[r] for r in drop} - keep_commits
    snaps = os.path.join(root, "snapshots"); blobs = os.path.join(root, "blobs")

    def links(commit):
        """The models--X/blobs entries one snapshot's files point at."""
        out = set()
        for dp, _, fs in os.walk(os.path.join(snaps, commit)):
            for f in fs:
                p = os.path.join(dp, f)
                if os.path.islink(p):
                    out.add(os.path.normpath(os.path.join(os.path.dirname(p), os.readlink(p))))
        return out

    kept = set().union(*(links(c) for c in os.listdir(snaps) if c not in drop_commits))
    gone = set().union(*(links(c) for c in drop_commits)) - kept if drop_commits else set()
    gone = {b for b in gone if os.path.dirname(b) == blobs}
    size = sum(os.path.getsize(b) for b in gone if os.path.isfile(b) and not os.path.islink(b))
    kept_refs = [r for r in drop if refs[r] in keep_commits]
    print(f"{a.repo}: {len(drop_commits)} snapshots, {len(gone)} blobs, {size / 1e9:.2f} GB"
          + (f"; kept (commit also under another ref): {kept_refs}" if kept_refs else ""), flush=True)
    if not a.delete:
        print("dry run; --delete to remove"); return
    for b in gone:
        os.remove(b)                                   # a regular file, or only the link into the shared store
    for c in drop_commits:
        shutil.rmtree(os.path.join(snaps, c))
    for r in drop:
        if refs[r] in drop_commits:
            os.remove(os.path.join(root, "refs", r))
    print("deleted")


if __name__ == "__main__":
    main()
