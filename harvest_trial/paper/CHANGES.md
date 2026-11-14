# Paper: pg's requests and the changes made

Every request pg makes about the paper, and what was changed in the doc for it.
Changes go into the doc as suggestions (pg accepts, rejects or rewrites each); nothing here is final.
Full doc: claude.ai/artifact/Trfk1oPddipAaoNgeAD7zQ. Short doc: claude.ai/artifact/RCmjwurHzoLUPmvviKfkfY.

## 14 Nov 2026

| # | pg asked | Change in the full doc | Suggestion | Short doc |
|---|---|---|---|---|
| 1 | Inclusion rule: "same fine tuning, as similar as possible pre training"; MiMo omitted (fine-tuning differs ~3x) | Method, "Which families qualify" paragraph | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 2 | Table of models tested and how much each qualifies | Table under that paragraph | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 3 | Columns: family/sizes, pre-training, fine-tuning, verdict; DeepSeek "strong - best fit", GPT "strong" | Table rebuilt | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 4 | Show it as an inclusion/exclusion table: which were considered and excluded | Verdict column says Included / Excluded with reason | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 5 | "still very clunky" | Cells cut to one or two words; MiMo's 3x RL and Claude's 59% run left to the text | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 6 | Second exclusion rule, his words: "not large enough to show any change of behaviour of care" | Last sentence of the paragraph | in sg_7t2cdurlby | sg_wcpvzqadrj (Method) |
| 7 | Report the Qwen size ladder as an adjunct: small models could only do the stated-choice test, not the harvest test | New 4.6 "Earlier evidence: what smaller models say" (Qwen2.5 0.5B-32B, Llama 70B vs 8B, follow-up D's wrong rule as the catch) | sg_caisvqzpyq | sg_b6e6db3kwf (short paragraph, end of "What we saw") |
| 8 | Fishing game (GovSim): not reported (copying, not care; capability claim withdrawn 18 Oct; pg stopped it twice) | none | - | - |
| 10 | "ok and the current copy": today's changes mirrored into the short version | - | - | sg_wcpvzqadrj, sg_b6e6db3kwf; plus pg's proposition (full doc sg_xwpnlhcmy5) as one paragraph in "Why it should be true", sg_mdibu3fyj6 |
| 11 | Subtitle: positive and forward ("a falsifiable result, with a forward direction for labs to test alignment"), not "what labs can do to break it"; no "The claim." box (papers don't start that way); a claim is general, never a finding or a game detail | Subtitle second sentence replaced; claim box deleted; abstract paragraph 2 opens with the general claim (intelligence expands what a mind recognises as worth preserving; alignment grows with intelligence) | sg_e3wl7kialp, sg_uzh3frfivj, sg_xazx87apc9 | sg_om1wnx6tph, sg_5r7ci2wm5s, sg_7ca0ycqoz1 |
| 12 | Subtitle in his words: "a falsifiable result on selected models with falsifiable direction for other frontier lab model testing"; claim in his words: values are proxies for a fundamental goal, clearer with intelligence; that goal is the same for humans, life and AI, Semataxy, decreasing uncertainty over time beyond the self; tested in simulated environments, agents of different intelligence acting towards other life and organisation against their own welfare. He worried it reads too philosophical | Subtitle and abstract paragraph 2 amended to his wording (kept, as the paper's thesis) | sg_e3wl7kialp, sg_xazx87apc9 (amended) | sg_om1wnx6tph, sg_7ca0ycqoz1 (amended) |
| 13 | "Values are proxies ... clearer to see" is the logic, not the claim; the claim is that the fundamental goal of humans, life and AI is the same, Semataxy | Abstract paragraph 2 reordered: the claim first, then "The logic is this." | sg_xazx87apc9 (amended) | sg_7ca0ycqoz1 (amended) |
| 14 | Is the claim paragraph the end of the abstract, or should the result and conclusion follow? | Abstract given the usual order (view, claim, logic, test, results, conclusion): "The test is simple" paragraph and the five-item findings list replaced by one results paragraph; last paragraph rewritten as the conclusion. The list's detail stays in section 4 | sg_8qdftcmcc3, sg_dsw4is5tls | short version's summary already has results and the ask; unchanged |
| 9 | Working rule: write changes straight into the doc as suggestions, don't ask; keep it fluid; log every change and request | this file | - | - |

Earlier suggestions still open in the full doc: sg_xwpnlhcmy5 (section 6, pg's proposition),
sg_351405449u (renumbering), sg_vhku4aa6z8 and sg_damakreiun (essay as Appendix A),
sg_vej3wlkaqs (Appendix A text).

Still open with pg: affiliation for the author line; headline "expresses/chooses" vs "finds it themselves";
section 1 point 5 "Superintelligence protects life" against the appendix's "does not promise kindness to every individual".
