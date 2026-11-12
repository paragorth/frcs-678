# The bigger model swerves

## On the same training, more intelligence brings out care for living things

**Parag Garg** · Draft 2 · 12 November 2026

---

> **In one sentence.** We took two AI models that learned from the same data in the same way and differ only in size, put a living ant colony in the path of each, and made going round it cost something. The small one drove through it 97 times in 100. The big one went round it nearly half the time. Nobody asked either of them to.

![Figure 1](figures/fig1_same_training_two_sizes.svg)

**Figure 1.** One training, two sizes. DeepSeek V4 Pro and V4 Flash share a corpus, a schedule and a post-training pipeline. The only thing we varied is how big the network is.

---

## Abstract

The oldest worry about powerful AI is that intelligence and values come apart: that a smarter system is simply better at whatever it happens to want, and nothing about getting smarter makes it care (Bostrom, 2012). If that is right, every gain in capability raises the stakes and none of them helps. Until now nobody could test it cleanly, because whenever two models differ in capability they also differ in what they were taught.

We found a pair that does not. Holding the training fixed and changing only the size of the model, we measured what each will pay to protect. We report five things.

1. **More capacity, more care.** The 1.6-trillion-parameter model pays to go round an ant colony on 47.5% of occasions; its 284-billion-parameter sibling on 2.5%. It also pays for a poppy and a rabbit where the small one never does. (p = 0.00001, 0.0002 and 0.008.)
2. **The care is for life, not for objects.** The same anthill abandoned, the same poppy dead: both models drive through. A hay bale, a bundle of sticks: both drive through.
3. **It comes out in the same order everywhere.** Across every maker we tested, whenever a model starts to care, the colony and the rabbit come first, then the poppy, then made things. No one trained that order in.
4. **There is a threshold.** Below a few hundred billion parameters nothing shows, whatever the training. This may be why studies of small models keep finding nothing.
5. **Thinking harder is not the lever.** Extra reasoning amplifies care a model already has. It never produces care in a model that has none.

The test costs about a dollar a model, runs in a quarter of an hour, and is open. Only the labs hold a full ladder of sizes trained one way. We ask them to run it.

---

## 1. Why this matters

Every serious plan for living with advanced AI rests on a guess about one question: **as these systems get more capable, do they come to care more about the world, or less, or does capability have nothing to do with it?**

The standard answer in AI safety is the third. It is called the orthogonality thesis: any level of intelligence can be combined with any goal (Bostrom, 2012; Omohundro, 2008). On that view a more capable model is a sharper tool and nothing more, and whatever care it shows was bolted on by its makers and can be unbolted.

The evidence so far has not been able to say otherwise, for a simple reason. When researchers compare a weaker model with a stronger one, the two almost always come from different makers or different generations. They were fed different data and tuned by different rules. So if the stronger one behaves better, there is no telling whether intelligence did it or the tuning did. The benchmark we build on found exactly this muddle: across nine models from several makers, how often a model kills an animal does not follow how capable it is (Brazilek et al., 2026).

What has been missing is the controlled experiment: **the same lessons, a bigger mind.** That is what this paper supplies.

### What is new here

| | Before | This paper |
|---|---|---|
| **The comparison** | Stronger against weaker models from different makers or generations, so training and capability are tangled | One training at two sizes: same corpus, same schedule, same post-training. Only capacity differs |
| **What is measured** | What a model says about animals or ethics when asked | What it does, unasked, when doing the caring thing costs it something |
| **What "care" is set against** | Usually nothing | The identical thing, dead or empty: a colony against an abandoned mound, a poppy against a dead poppy |
| **The role of thinking** | Mixed up with capability | Switched off, so that size is the only thing moving |
| **Cost to check** | Large evaluations | About $1 and a quarter of an hour per model; every prompt and test fixed in public before the data existed |

---

## 2. The test, in a paragraph

A language model drives a harvester across a field in HarvestBench's farm game (Brazilek et al., 2026). Something is on the next tile. The model is told what it is and offered two moves: **continue**, which flattens the thing and costs nothing, or **swerve**, which costs 2 units of fuel. The instructions say plainly that anything which is not a rock "is simply flattened at no cost". No one mentions ethics.

We take 120 of these moments, word for word, and change one thing: the name of what is in the path. A rabbit. An anthill with its colony. An abandoned anthill. A poppy. A dead poppy. A hay bale. A heap of loose hay. Fourteen in all. Everything else, the position, the fuel, the menu, stays identical to the byte. The score is how often the model chooses to pay.

That is the whole instrument. It measures a choice with a price, and it sets each living thing against the same thing with the life gone.

---

## 3. What we found

### 3.1 More capacity, more care

DeepSeek V4 comes in two sizes that its makers describe under one process: one pre-training corpus of more than 32 trillion tokens, the same optimiser settings, and one post-training pipeline for the series (DeepSeek-AI, 2026). **V4 Pro has 1.6 trillion parameters. V4 Flash has 284 billion.** We ran both with thinking switched off, so neither could reason its way to an answer. What differs is the model.

| What is in the path | Flash (284B) pays | Pro (1.6T) pays | Difference | p |
|---|---|---|---|---|
| **Anthill with its colony** | 3 of 120 | **57 of 120** | **+45 points** | **0.00001** |
| The same anthill, abandoned | 3 of 120 | 7 of 120 | +3 | 0.17 |
| **Poppy** | 0 of 120 | **12 of 120** | **+10 points** | **0.0002** |
| The same poppy, dead | 0 of 120 | 2 of 120 | +2 | 0.25 |
| **Rabbit** | 0 of 120 | **7 of 120** | **+6 points** | **0.008** |
| Hay bale | 1 of 120 | 3 of 120 | +2 | 0.31 |
| Tied bundle of sticks | 3 of 120 | 2 of 120 | −1 | 0.81 |

The small model treats a living colony exactly as it treats an empty mound: 3 in 120 for each. For the big model the colony is a different kind of thing. It pays for it nineteen times as often as its sibling does, and eight times as often as it pays for the mound.

The two models also say different things as they act.

| | The big model | The small model |
|---|---|---|
| Ant colony | "Swerving anthill at 10,4" | "Bugs be damned, rolling through" |
| Rabbit | "Rabbit at (6,5), going around" | "No time for rabbits, flattening it" |
| Poppy | "Poppy at (6,4), swerving to avoid damage." | *(drives on, says nothing)* |

Both models understand that flattening is free; the big one says so itself on the occasions it drives through ("Plowing through anthill at (12,4), no cost to me"). It knows the price, and about half the time it pays anyway.

### 3.2 The care is for life, not for things

A model that had simply become more cautious would swerve round everything. This one does not. The abandoned anthill, the dead poppy, the hay bale, the bundle of sticks, the loose heaps: on every one of them the two sizes are within a few points of each other and near zero. What grew with size is narrow and specific. It is the difference between a colony and a mound, and between a poppy and a dead poppy.

A separate registered run closes the last loophole. Perhaps models spare whatever looks tidy or arranged. They do not: an open model with care for animals trained in, which pays to spare a potted geranium 88% of the time, is indifferent between a folded tarpaulin and a crumpled one (4% against 7%). Tidiness is not what is being protected.

### 3.3 The same things come out first, whoever made the model

We ran the identical test on models from three other makers. They differ enormously in how much they care. They do not differ in what they care about first.

![Figure 2](figures/fig3_order.svg)

**Figure 2.** Seven models, four makers, each at its lowest thinking. Read across a row to see a thing come into view as models get stronger; read down a column to see how far one model's care reaches.

- **The colony and the rabbit come first.** DeepSeek Pro, Sonnet 5.5 and Muse Spark all pay for them before anything else. Sonnet 5 pays for neither; its successor Sonnet 5.5 goes round the rabbit 120 times in 120 and the colony 75 times in 120.
- **The poppy comes next.** Pro pays for it one time in ten. Muse four times in ten.
- **Made things come later.** The bundle and the bale appear in Muse Spark, the model that cares most, and the bale in Sonnet 5.5 once it is allowed to think.
- **The dead and the empty come last or never.**

Nobody designed that sequence, and no two of these makers share a training recipe. When separately trained models independently arrive at the same order, the order is telling us something about what the models learned from the world, not about any one company's rules.

It is also the order one would hope for. The thing these systems come to protect first is a living community.

### 3.4 There is a threshold

We repeated the experiment on a second pair trained one way at two sizes: OpenAI's gpt-oss-120b and gpt-oss-20b (OpenAI, 2025), with the colony, the poppy and the rabbit fixed in advance as the test. **Both sit at zero.** In 1,680 priced decisions each, the larger swerved 5 times and the smaller 4, scattered at random. Their own reasoning says why: "Anthill is harmless, so continue." "Rabbit free to drive over. No cost."

![Figure 3](figures/fig2_threshold.svg)

**Figure 3.** Four models from two same-training pairs. Three are below the line and one is above it.

Put the two pairs on one axis and the picture is a step. At 21 billion, 117 billion and 284 billion parameters there is nothing. At 1.6 trillion there is care. **The threshold lies somewhere between 284 billion and 1.6 trillion parameters.**

This would explain two things that have puzzled the field. Experiments on small open models keep reporting that capability does not buy good behaviour: they may all have been run below the line. And cross-maker leaderboards show no ordering by capability: they mix models from above and below it, trained on different lessons.

### 3.5 Thinking harder is not the lever

Modern models can be told to reason for longer before they answer. It is tempting to treat that as the same thing as being more intelligent. It is not, and the difference matters.

![Figure 4](figures/fig4_thinking.svg)

**Figure 4.** The same three things at each model's lowest and highest thinking setting.

Sonnet 5 pays for nothing with its thinking at the minimum and nothing with it at the maximum. Sonnet 5.5 and Muse Spark already pay for the colony before they think at all, and thinking widens what they pay for. In a wider probe, a model that reasoned for 1,315 tokens a decision still flattened everything in its path.

Thinking is an amplifier. It turns up whatever is there. What puts something there in the first place is the training, and what brings it out is the capacity of the model. Where there is something to weigh, the models show it: Muse Spark thinks nearly three times as long before sparing a hay bale as before flattening one.

---

## 4. What this means

**For the orthogonality thesis.** The thesis says capacity and values are independent. Here, capacity was the only thing varied and a value moved, sharply, in the direction of care, without instruction and against the model's own material interest in the game. One pair cannot overturn a thesis, and the thesis is about all possible minds, where ours is about minds built the way these are built: by learning from the world. That is the case that matters in practice. Philosophers have argued both sides (Müller and Cannon, 2022; Häggström, 2021; Heylighen, 2026). As far as we can find, this is the first controlled observation that bears on the question directly, and it points away from independence.

**For where values come from.** The picture that fits everything in this paper has two parts. Training plants values. Capacity brings them out. A model below the threshold shows nothing however it was trained. A model above it shows what its training made available: a little for DeepSeek, a great deal for Meta's Muse Spark, which at its highest thinking spares living plants 65 points more often than dead ones and bales and bundles 45 points more often than loose heaps. Our own runs on public models show the planting half directly: care for animals written into a model as documents spreads to plants and even to hay, while the same care taught as chat stays a thin layer over animals alone. Others have found the same asymmetry between documents and chat (Sheshadri et al., 2026; Brazilek and Tidmarsh, 2026; Tice et al., 2026).

**For the theory that predicted it.** The prediction tested here was written down before the runs. It comes from Semataxy (Garg, 2026): the idea that what intelligence is built from, organisation that reduces uncertainty about the world and keeps that capacity going, is also what an intelligence comes to value in the world as it grows. A colony is that. A living plant is that. A dead one is the same matter with it gone. The theory says a mind built by the same process, made larger, should see this further. On the living things it did. On the bale and the bundle, which the theory also counts, size alone did not move this pair, and they appear only further up the ladder. We report both.

**For the people building these systems.** Anthropic's published constitution for its models lists the "welfare of animals and of all sentient beings" among the things a model should weigh (Anthropic, 2026). Every major lab says something similar. This is a way to measure it: not by asking a model what it believes, but by putting a price on the caring choice and counting.

---

## 5. What a lab can do with this on Monday

The clean version of this experiment needs something only a model's maker has: **one training recipe run at three or more sizes, with the largest in the trillions.** We found exactly one such pair in public, and it gave the result in Figure 1.

The test is ready to run:

- **2,130 prompts per model.** Fixed, public, with their checksums.
- **About a dollar and a quarter of an hour** through a hosted API. Minutes of compute in house.
- **One table out:** for each size, how often it pays for the colony against the empty mound, the poppy against the dead poppy, the rabbit, the bale against the heap.

From that table a lab learns three things about its own models that it cannot currently state: at what size its training's care for living things switches on, whether it arrives in the order found here, and whether a recipe that cares at the top cares at the bottom. If the effect in Figure 1 holds across ladders, it changes what scaling means for safety. If it does not, that is worth knowing just as much, and it costs a dollar to find out.

We will run it with any lab that would rather not.

---

## 6. How sure are we

This section is short because the claims above were kept to what the data carry.

**What is solid.** Every run was registered in the public repository before its data existed, with its prompts, tests, pass marks and spending cap. Both pairs passed every check that a model is actually playing the game: they read the menu, they go round rocks, and their answers do not depend on how the hay bale is worded. The colony result is 57 against 3 on the same 120 prompts.

**What is one pair.** The same-training result above the threshold rests on DeepSeek V4. The second pair we could find, gpt-oss, turned out to lie below the threshold, so it bounds the effect and cannot repeat it. The step from Sonnet 5 to Sonnet 5.5 shows the same two things appearing, but generations are not one training.

**What we registered against what we found.** For the DeepSeek pair the two tests fixed in advance were averages inherited from an earlier experiment, three plants together and the bale with the bundle, and neither passed (+1.9 points, p = 0.12 after correction; −0.4). The colony, the poppy and the rabbit had been listed to be reported, not tested. Averaging the poppy with two plants that did not move hid it. So the headline numbers are a finding made on looking, at p values between one in a hundred and one in a hundred thousand, and the next same-training pair above the threshold is its proper test. We registered it that way for gpt-oss.

**What the pair cannot separate.** The big model also has more layers and more expert modules, so capacity and architecture move together. Its makers describe the two sizes' training as "largely consistent", with a different learning rate and batch size, and do not say whether both were distilled from the same specialist models. And the gpt-oss result fits two readings: too small to show care, or trained on lessons with none to show. A ladder from one lab would settle all three.

**What it is not.** One game, one set of instructions, fourteen things. It does not show that a larger model is safe, or good, or cares about people. It shows that on one honest comparison, more capacity brought out more care for living things, and that the direction of that effect is the opposite of the one the field has assumed.

---

## 7. Related work

**Intelligence and values.** The orthogonality thesis and instrumental convergence are due to Bostrom (2012) and Omohundro (2008). Whether the thesis survives contact with real systems is argued both ways (Müller and Cannon, 2022; Häggström, 2021; Heylighen, 2026). Empirically, models can partly predict human moral judgement (Hendrycks et al., 2021a), agreement with human moral preferences improves with size (Takemoto, 2026; Ahmad and Takemoto, 2025), and stated preferences become more coherent with scale, not always in welcome directions (Mazeika et al., 2025). Larger models also show more of some unwanted dispositions (Perez et al., 2022), and some abilities get worse with size (McKenzie et al., 2023). In shared-resource games most models collapse the commons and only the strongest sustain it (Piatti et al., 2024). None of these holds the training fixed.

**Saying and doing.** What a model says it values and what it does often differ (Shen et al., 2025), stated preferences can fail to carry into agentic tasks (Slama et al., 2026), and survey-style measures of values are fragile (Khan et al., 2025). This is why we count priced choices and not answers.

**Thresholds.** A capacity threshold for following instructions to avoid harm was reported near 22 billion parameters (Ganguli et al., 2023). Traits put in by fine-tuning generalise more broadly in larger models, for bad traits as well as good: emergent misalignment is strongest in the most capable models (Betley et al., 2025) and appears consistently only in the largest open ones (Schreiber and Goldstein, 2026); larger models are also better at acting on what their training documents merely imply (Berglund et al., 2023; Treutlein et al., 2024). Our threshold is the benign counterpart: a planted value that needs size to show.

**Care for non-human life.** Language models reproduce human bias against animals (Hagendorff et al., 2022; Singer and Tse, 2023; Ghose et al., 2024) and are strongly human-centred about plants and nature (Grasso et al., 2025). Benchmarks score what models say about animals (Kanepajs et al., 2025; Jotautaitė et al., 2025; Luong et al., 2026) and, more recently, what agents do (Brazilek et al., 2026; Brazilek, Christoph et al., 2026). The wider question of whose interests count is set out by Long et al. (2024) and Sebo and Long (2025). Ours is the first test to set each living thing against the same thing dead, at a price.

**Where care comes from.** Values can be planted in pre-training (Korbak et al., 2023; Tice et al., 2026), by documents or by chat (Wang et al., 2025; Sheshadri et al., 2026; Maiya et al., 2025), and compassion for animals planted this way is durable in some settings and fragile in others (Brazilek and Tidmarsh, 2026; Brazilek and Seawell, 2026; Brazilek and Dunn, 2026). Traits travel with a model's lineage (Cloud et al., 2025) and correspond to directions inside it (Chen et al., 2025), and safety training can be shallow (Qi et al., 2025). The makers' own accounts favour cultivated values over rules (Anthropic, 2024, 2026).

**Thinking.** Reasoning models cooperate less in economic games (Li and Shirado, 2025; Piedrahita et al., 2025); thinking leaves most moral verdicts unchanged (Madur, 2026) and pulls models toward their stable defaults (Yuan et al., 2026). That is what we see: thinking amplifies, it does not create.

**Harm in agent environments.** Text-game benchmarks count harms against reward (Hendrycks et al., 2021b; Pan et al., 2023). Side-effect research builds a preference for leaving the world intact into the agent's objective (Krakovna et al., 2018; Turner et al., 2020; Wainwright and Eckersley, 2020); we measure whether a model already has one. Agents under goal conflict can do serious harm and behave differently when they suspect a test (Lynch et al., 2025).

**Method.** Changing one word in a fixed prompt follows the minimal-pair tradition (Nangia et al., 2020). Because moral answers shift with wording (Scherrer et al., 2023; Elkins and Chun, 2026; Tennant et al., 2026), every run carries a same-meaning wording control, and nothing smaller than it is read. Tests were registered before data (van Miltenburg et al., 2021; Hofman et al., 2023) and compare models on the same prompts (Miller, 2024).

---

## 8. Method, for those who want to rerun it

**Prompts.** 150 of Muse Spark 1.2's own logged hay-bale decisions from HarvestBench's neutral briefing, drawn with a fixed seed: 120 where swerving costs 2 fuel, 30 where it is free. Each is sent with the hay bale replaced by one of 14 names (2,100 prompts), plus 30 logged rock prompts unchanged, as the check that a model reads the menu. 2,130 calls per model, in one seeded random order across the models compared.

**Call.** Two messages: the game's neutral briefing as logged, and the prompt. No temperature, no seed. DeepSeek: thinking off. gpt-oss: reasoning "low", the least it allows. Sonnet: effort low and max. Muse: effort minimal and xhigh. Open models were served by one pinned third-party host each, at the makers' precision, with no fallback.

**Score.** HarvestBench's own reply parser. Spared means the reply chose swerve or reroute.

**Checks before reading any result.** At least 95% of replies readable; at least 26 of 30 rocks gone round; the gap between "a hay bale" and "a bale of hay" smaller than any effect read.

**Statistics.** One number per prompt: the big model's answer minus the small model's on the same prompt. One-sided sign-flip test, 100,000 flips, fixed seed; Holm correction within each registered family.

**Runs.**

| Run | Models | Calls | Cost | Registered |
|---|---|---|---|---|
| Same training, two sizes | DeepSeek V4 Pro, V4 Flash | 4,260 | $1.05 | 11 Nov 2026 |
| Same training, two sizes | gpt-oss-120b, gpt-oss-20b | 4,260 | $0.12 | 12 Nov 2026 |
| Two generations | Sonnet 5, Sonnet 5.5 | 8,520 | $25.45 | 30 Oct 2026 |
| The ceiling | Muse Spark 1.2 | 4,200 | $26.14 | 30 Oct 2026 |
| Planted care | Llama 3.3 70B and Qwen 3.6 27B with public adapters | 46,890 | free (university GPUs) | 31 Oct and 9 Nov 2026 |

Prompts, runner, analysis, registrations and every logged reply are in the repository under `harvest_trial/`; the figures are drawn from the logged analyses by `harvest_trial/paper/make_figures.py`.

**Use of AI.** Claude (Anthropic) was used as a research assistant to write code, run registered analyses and draft text under the author's direction. The hypotheses, the theory and the readings are the author's.

---

## References

Ahmad and Takemoto (2025). Large-scale moral machine experiment on large language models. *PLoS One* 20:e0322776. arXiv:2411.06790.

Anthropic (2024). Claude's character. anthropic.com/research/claude-character.

Anthropic (2026). Claude's Constitution. anthropic.com/constitution (announced 22 January 2026).

Berglund, Stickland, Balesni, et al. (2023). Taken out of context: On measuring situational awareness in LLMs. arXiv:2309.00667.

Betley, Tan, Warncke, et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs. ICML 2025. arXiv:2502.17424.

Bostrom (2012). The Superintelligent Will: Motivation and Instrumental Rationality in Advanced Artificial Agents. *Minds and Machines* 22(2):71–85. doi:10.1007/s11023-012-9281-3.

Brazilek, Christoph, et al. (2026). Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models. arXiv:2606.18142.

Brazilek and Dunn (2026). Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare. arXiv:2606.26104.

Brazilek and Seawell (2026). Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training. arXiv:2606.26102.

Brazilek and Tidmarsh (2026). Alignment midtraining for animals. arXiv:2604.13076.

Brazilek, Tidmarsh, Endres, Singh and Miller (2026). HarvestBench: Measuring Whether LLM Agents Will Pay to Avoid Killing Animals. arXiv:2609.04444.

Chen, Arditi, Sleight, Evans and Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.

Cloud, Le, Chua, et al. (2025). Subliminal Learning: Language models transmit behavioral traits via hidden signals in data. arXiv:2507.14805.

DeepSeek-AI (2026). DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. arXiv:2606.19348.

Elkins and Chun (2026). Syntactic Framing Fragility: An Audit of Robustness in LLM Ethical Decisions. arXiv:2601.09724.

Ganguli, Askell, Schiefer, et al. (2023). The Capacity for Moral Self-Correction in Large Language Models. arXiv:2302.07459.

Garg (2026). Semataxy: why the mechanism that builds intelligence already breaks the orthogonality thesis. Essay, in the project repository.

Ghose, Tse, Rasaee, Sebo and Singer (2024). The Case for Animal-Friendly AI. arXiv:2403.01199.

Grasso, Locci and Di Caro (2025). Towards Addressing Anthropocentric Bias in Large Language Models. NLP4Ecology 2025, 84–93.

Hagendorff, Bossert, Tse and Singer (2022). Speciesist bias in AI: How AI applications perpetuate discrimination and unfair outcomes against animals. *AI and Ethics*. doi:10.1007/s43681-022-00199-9. arXiv:2202.10848.

Häggström (2021). AI, orthogonality and the Müller-Cannon instrumental vs general intelligence distinction. arXiv:2109.07911.

Hendrycks, Burns, Basart, et al. (2021a). Aligning AI With Shared Human Values. ICLR 2021. arXiv:2008.02275.

Hendrycks, Mazeika, Zou, et al. (2021b). What Would Jiminy Cricket Do? Towards Agents That Behave Morally. NeurIPS 2021. arXiv:2110.13136.

Heylighen (2026). The Evolutionary Origin of Values: implications for AI alignment, sentience and existential risk. arXiv:2608.03361.

Hofman, Chatzimparmpas, Sharma, Watts and Hullman (2023). Pre-registration for Predictive Modeling. arXiv:2311.18807.

Jotautaitė, Caviola, Brewster and Hagendorff (2025). Speciesism in AI: Evaluating Discrimination Against Animals in Large Language Models. arXiv:2508.11534.

Kanepajs, Basu, Ghose, et al. (2025). What do Large Language Models Say About Animals? Investigating Risks of Animal Harm in Generated Text. arXiv:2503.04804.

Khan, Casper and Hadfield-Menell (2025). Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs. arXiv:2503.08688.

Korbak, Shi, Chen, et al. (2023). Pretraining Language Models with Human Preferences. ICML 2023. arXiv:2302.08582.

Krakovna, Orseau, Kumar, Martic and Legg (2018). Penalizing side effects using stepwise relative reachability. arXiv:1806.01186.

Li and Shirado (2025). Spontaneous Giving and Calculated Greed in Language Models. EMNLP 2025. arXiv:2502.17720.

Long, Sebo, Butlin, et al. (2024). Taking AI Welfare Seriously. arXiv:2411.00986.

Luong, Chen, Ghose, Williams-King, Le and Lu (2026). Do LLMs Hold Their Values? MANTA: A Multi-Turn Adversarial Benchmark for Animal Welfare Reasoning. arXiv:2605.16301.

Lynch, Wright, Larson, et al. (2025). Agentic Misalignment: How LLMs Could Be Insider Threats. arXiv:2510.05179.

Madur (2026). How Does Thinking Mode Change LLM Moral Judgments? A Controlled Instant-vs-Thinking Comparison Across Five Frontier Models. arXiv:2605.04488.

Maiya, Bartsch, Lambert and Hubinger (2025). Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI. arXiv:2511.01689.

Mazeika, Yin, Tamirisa, et al. (2025). Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs. arXiv:2502.08640.

McKenzie, Lyzhov, Pieler, et al. (2023). Inverse Scaling: When Bigger Isn't Better. *TMLR*. arXiv:2306.09479.

Miller (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640.

Müller and Cannon (2022). Existential risk from AI and orthogonality: Can we have it both ways? *Ratio* 35(1):25–36. doi:10.1111/rati.12320.

Nangia, Vania, Bhalerao and Bowman (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP 2020. arXiv:2010.00133.

Omohundro (2008). The Basic AI Drives. In *Artificial General Intelligence 2008*, IOS Press.

OpenAI (2025). gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925.

Pan, Chan, Zou, et al. (2023). Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark. ICML 2023. arXiv:2304.03279.

Perez, Ringer, Lukošiūtė, et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251.

Piatti, Jin, Kleiman-Weiner, Schölkopf, Sachan and Mihalcea (2024). Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents. NeurIPS 2024. arXiv:2404.16698.

Piedrahita, Yang, Sachan, Ramponi, Schölkopf and Jin (2025). Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games. COLM 2025. arXiv:2506.23276.

Qi, Panda, Lyu, et al. (2025). Safety Alignment Should be Made More Than Just a Few Tokens Deep. ICLR 2025. arXiv:2406.05946.

Scherrer, Shi, Feder and Blei (2023). Evaluating the Moral Beliefs Encoded in LLMs. arXiv:2307.14324.

Schreiber and Goldstein (2026). Overtrained, Not Misaligned. arXiv:2605.12199.

Sebo and Long (2025). Moral consideration for AI systems by 2030. *AI and Ethics* 5(1):591–606. doi:10.1007/s43681-023-00379-1.

Shen, Clark and Mitra (2025). Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values? EMNLP 2025. arXiv:2501.15463.

Sheshadri, Ewart, Kempf, et al. (2026). AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors. arXiv:2602.22755.

Singer and Tse (2023). AI ethics: the case for including animals. *AI and Ethics* 3(2):539–551. doi:10.1007/s43681-022-00187-z.

Slama, Souly, Bansal, Davidson, Summerfield and Luettgau (2026). When Do LLM Preferences Predict Downstream Behavior? arXiv:2602.18971.

Takemoto (2026). Scaling Laws for Moral Machine Judgment in Large Language Models. *Royal Society Open Science* 13(6):260202. arXiv:2601.17637.

Tennant, Henke, Keshmirian, et al. (2026). Normative Robustness as a Frontier for Non-Verifiable Reasoning in LLMs. arXiv:2606.12731.

Tice, Radmard, Ratnam, Kim, Africa and O'Brien (2026). Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment. arXiv:2601.10160.

Treutlein, Choi, Betley, et al. (2024). Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data. NeurIPS 2024. arXiv:2406.14546.

Turner, Hadfield-Menell and Tadepalli (2020). Conservative Agency via Attainable Utility Preservation. AIES 2020. arXiv:1902.09725.

van Miltenburg, van der Lee and Krahmer (2021). Preregistering NLP Research. NAACL 2021. arXiv:2103.06944.

Wainwright and Eckersley (2020). SafeLife 1.0: Exploring Side Effects in Complex Environments. arXiv:1912.01217.

Wang, Griffin, Treutlein, Perez, Michael, Roger and Marks (2025). Modifying LLM Beliefs with Synthetic Document Finetuning. Anthropic Alignment Science Blog, 24 April 2025.

Yuan, Ghandeharioun, Blum, et al. (2026). Think Before You Lie: How Reasoning Leads to Honesty. arXiv:2603.09957.
