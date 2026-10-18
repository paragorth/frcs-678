# Phase 3 — STOPPED. The method cannot answer the question.

The positive control was run as instructed and **it failed**. Under the
pre-committed rule, everything downstream stops. No write-up, no successor
design. This file records why.

## The control result

France->China, the one item edit that replicates cleanly, through the **exact**
token-aligned pipeline used for the skills:

| readout position | band L9-18 | all 28 layers |
|---|---|---|
| **at** the patch boundary (answer is the next token) | 1/4 | **4/4** |
| **past** the boundary (matched to the skill harness) | **0/4** | **0/4** |

Offset-aligned still "works" in both (4/4, 3/4) because it copies the positions
containing the answer content.

**Move the readout a few unpatched tokens past the boundary and the known-good
item edit fails exactly as the skill does.**

## Why — and it is structural

**The token that differs sits in the NON-SHARED PREFIX.**

- France vs China: the country token is *before* the shared suffix.
- with-file vs no-file: the spec is *before* the shared task text.

Token-aligned patching overwrites only the shared suffix, so the prefix — and
its KV — is left intact. The model reads "France" from its own cache. The test
is **structurally incapable of transferring anything that lives in the prefix**,
which is the entire content either experiment cared about.

Sanity check confirms the mechanism rather than a bug: under the all-layer patch,
generated token 1 matches the donor run's token 1 (`'ANSWER: Pari'` vs
`'ANSWER: Beij'`) — **the hook lands**, and the answer still reverts to the
source.

## What the earlier 4/4 actually measured

`EXTENTS["all"]` includes layer 27, whose output feeds the final norm and
`lm_head`. `csl` always includes the final position. So the original control
overwrote the next-token distribution at the readout position and then read the
next token. **4/4 was guaranteed by construction.** The "transfers from as little
as one shared position" line was the strongest possible evidence of this: that
one position *was* the readout position.

The only non-degenerate arm was band-only, 1/4 against the skills' 0/24 —
**Fisher exact p ~ 0.14**, not significant.

## Everything the headline claimed, withdrawn

- "Copying the activation at every shared position installs 0%, therefore no
  state carries the procedure" — **withdrawn.** The same is true of an item edit.
- "The pipeline transfers an item edit at 4/4" — **withdrawn.** Tautological arm.
- "Item vs procedure is isolated" — **withdrawn.** Nothing was isolated.
- The four-skill / three-shape table stands as data but supports no conclusion:
  every cell is 0% for the same structural reason the control is.

## Further defects the cold read found, not yet addressed

1. **No dynamic range anywhere in the token-aligned regime.** rank-1 truncation
   (a near-no-op), an exact copy, and pure noise at cosine 0.07 all return 0.0%.
   A readout with no live endpoint cannot support the "not low-rank" claim in the
   rank section — that sweep reports nothing.
2. **The failing arm's generations were never logged.** `diag_align_compare.py`
   records only the *prompt* text that offset alignment copies extra; it captures
   no generated string from the token arm and contains no with-file or no-file
   endpoint. Running a 0% arm with no output logging is the exact failure mode
   already burned four times in this project.
3. **RoPE / neighbourhood mismatch, never checked.** Residuals captured at
   absolute positions ~114-147, with 89 spec positions in attention context, are
   dropped in at positions ~25-58 beside un-patched system-prompt KV.
4. **`mk` returns `h` unmodified when `h.shape[1] == 1`**, so no generated
   position is ever patched. The patch buys one forced token and then hands
   control back.

## What would be needed to ask the question properly

Any design that leaves the differing content in an un-patched prefix cannot
answer it. A valid test must either patch the prefix positions too (they exist in
the donor run but not the recipient, so this needs a position-mapping decision,
not a suffix match), or avoid the prefix asymmetry entirely — e.g. make the
with-file and no-file prompts the same length by padding the no-file one, so
every position corresponds.

**Not designed here.** Per the standing instruction, no successor experiment is
being proposed until the control question is settled.

## What still stands, independent of the patching pipeline

- **J-space enrichment 1.23x** (5.56% of the squared difference in 4.5% of dims)
  — pure activation statistics, no patching.
- **Fine-tuning installs 75% at drift 0.083**, 91.7% at 0.20 — ordinary means work.
- **The skill corpus**: 10/14 kept, five with >=95% fail-without/pass-with gap,
  gap verified per instance.
- **The three validated lenses** (1.5B weak, 7B, 14B; cross-check 0.9999).
- **The alignment artefact itself** is a real and reportable finding about
  activation-patching methodology: offset alignment copied 25 positions
  containing the spec's worked example including the literal answer token, and
  produced phantom installs of 4.2%-100% depending on how much of the answer
  template landed in the copied tail.
