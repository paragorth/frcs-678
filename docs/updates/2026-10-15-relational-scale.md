# Update, 15 October 2026: relational Semataxy, a measure suggested by art

**Idea.** A work's structure exists for an observer who can see it (`results/art/art.csv`). So measure a
model by how much of the world's structure it sees, not by its weights alone. Cut real text
(WikiText-103 test, 24 windows of 512 tokens) into blocks of b = 1..256 tokens, shuffle the blocks, and
record the rise in the model's loss. Shuffling keeps every token, so entropy is unchanged; only
organisation above scale b is destroyed. Tool: `tools/relational_scale.py`, laptop CPU.

**Result.** Unlike the weight measure, it orders the fixed-size Qwen generations correctly.

| Comparison (mean rise, blocks 8 to 128) | difference | standard error | windows where it rises |
|---|---|---|---|
| Qwen1.5 0.5B to Qwen2 0.5B | +0.024 | 0.011 | 15/24 |
| Qwen2 0.5B to Qwen2.5 0.5B | +0.037 | 0.008 | 19/24 |
| Qwen2.5 0.5B base to instruct | -0.001 | 0.005 | 13/24 |
| Qwen2.5 0.5B to 1.5B (instruct) | +0.045 | 0.011 | 20/24 |

Pythia 160M is lowest at every scale above 4 tokens and stops seeing structure one octave earlier.

**It is not simply the loss.** Instruct tuning raises base loss on this text (2.72 to 2.81) and leaves
the shuffle sensitivity unchanged. Across the generations, though, lower loss and higher sensitivity
move together, so loss is not fully separated yet.

**Caveats.** Small effects; one text domain; WikiText may be in the training data; three generations
is a small sample. Next: post-cutoff text, more domains, more windows, the 7B same-size set on CSD3
(Qwen1.5/2/2.5, Mistral, Llama, Gemma, DeepSeek distill), and a check controlling for base loss.
