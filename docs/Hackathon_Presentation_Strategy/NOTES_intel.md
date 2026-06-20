# Inference-Time Compute Hackathon #3: Judge & Strategy Intelligence

## Judge Profiles: The Five Decision-Makers

### Anthropic (Claude)
**What They Care About:** Real-world impact on specific, named users with domain expertise. Anthropic explicitly rejects "flashy interface, impressive prompt chain, 'look what it can do' projects." They want agentic systems solving concrete problems that could *only* be built with Claude's extended thinking, multi-agent orchestration, or long-context capabilities.

**What Impresses:** Domain expertise (doctors, attorneys, teachers) beats pure coding ability in their winning submissions across Opus 4.6–4.7 hackathons. Projects demonstrate eval-first development (build rubrics before features), spec-first planning (4+ hours before coding), and leverage Claude-specific strengths that would fail on smaller/cheaper models. Winning framing: "A tool answering in one sentence — who is this for, and what does their day look like without it?"

**Winning Signal:** Multi-step autonomous workflows with verifiable real-world impact. One-sentence answer: "This is for [named professional type] solving [specific problem]." Examples: CrossBeam (housing permits for attorneys), MedKit (voice clinical simulator), Maieutic (CS students explaining code). Healthcare, education, legal access, infrastructure, and safety are favored domains.

---

### Etched (Sohu Chip)
**What They Care About:** Applications where inference speed isn't just faster—it's *transformative*. Etched has bet their company on transformers going mainstream. CEO Gavin Uberti: "If transformers stick around, we become one of the biggest companies of all time." They want to see use cases that are economically infeasible on H100s but viable on Sohu's 21x faster throughput and 10x cheaper cost.

**What Impresses:** Real-time interactive world models (20+ FPS), agent swarms (10+ simultaneous agents with separate KV caches), sub-100ms voice response at full quality, parallel code search over many paths. The test: "Would this work on a CPU? If yes, it won't impress Etched." The inverse test: "Why is this economically infeasible on H100s but viable on Sohu?" If you can answer that, you're speaking their language.

**Winning Signal:** Make speed *visible* and *essential*. Put a latency counter or FPS counter in the demo UI. Explicitly close with: "On Etched's Sohu chip, this gets 21x more users at the same cost" or "21x more users at same cost." References to Oasis demo (Minecraft at 20 FPS, no game engine) show you understand their thesis.

---

### Cognition AI (Devin)
**What They Care About:** Algorithmic novelty in inference-time compute allocation—specifically, how agents spend compute at inference time. They published research (Kevin-32B, May 2025) proving that 8 sequential refinement steps with executable feedback beats 64 parallel samples at fixed compute budget. "Serial depth beats parallel breadth."

**What Impresses:** Novel approaches to compute allocation beyond naive parallelism. Agents with verifiable, executable feedback loops (code that runs/passes tests, math with formal proofs). Context-coherent agents maintaining state across long tasks. FrontierCode benchmark validation (code a maintainer would actually merge, not just tests passing). Process Reward Model scoring on intermediate steps. They explicitly warn against naive multi-agent parallelism without strong context sharing.

**Winning Signal:** Quantitative benchmarks on verifiable domains. Demonstrate serial refinement outperforming parallel sampling at fixed compute. Apply their Kevin-32B research to a new domain and cite it: "We applied your research on serial refinement to [domain] and found [result]." Show an agent catching its own bugs through executable feedback. Frame: "Serial refinement achieves X% vs. Y% parallel sampling at identical token budget."

---

### Mercor (APEX, Monty, Era of Evals)
**What They Care About:** Measurable, quantifiable outcomes with explicit baselines. CEO Brendan Foody frames it: "Evals are the new PRD." Mercor runs 10,000+ AI interviews/day via Monty at 700ms end-to-end latency. They've built the APEX benchmark (400 expert-crafted evaluation cases across finance, consulting, law, medicine).

**What Impresses:** Systems producing verifiable results under real-world conditions. Explicit baseline comparisons. For Real-Time track: beat Monty's 700ms and say so explicitly. For Agents: show task completion rate on a verifiable benchmark. For Talent Marketplace (Track 3): show correlation with human expert ratings on APEX dataset. Novel autograders, preference rankers, human-in-the-loop evaluation frameworks directly impress Foody.

**Winning Signal:** Put a number in your demo UI. "We achieve 47% on FrontierCode Diamond vs. 13% baseline." "Our voice AI responds in 289ms vs. Monty's 700ms." Track 3 signal: "This is an autograder that could plug into APEX and correlate with human expert ratings at [correlation coefficient]."

---

### Prime Intellect (Distributed Training, INTELLECT Models, Lab)
**What They Care About:** Projects that use 8x H100s in architecturally interesting ways—distributed inference, parallel rollouts, online RL fine-tuning during the hackathon, verifiable agentic systems with RL environments. CEO Vincent Weisser: intelligence should become "too cheap to meter." CTO Johannes Hagemann warns against a single superintelligence.

**What Impresses:** Actual model improvement during the hackathon. They launched Lab (May 2026)—a hosted post-training loop for self-improving agents—specifically for this hackathon use case: take a model, specialize it on your domain using RL, demonstrate measurable improvement in 24 hours. Using H100 allocation for training loops, not just inference serving. Building RL environments with verifiable rewards.

**Winning Signal:** Show a training/improvement curve from hour 1 to hour 23. "We didn't just deploy a model—we improved it." If you use their Lab platform, say so explicitly. Frame: "Accuracy at hour 1: 23%. Accuracy at hour 18 after RL fine-tuning: 52%. Here's the live RL environment we built and the training curve."

---

## Three Winning Strategy Patterns

### Pattern 1: Real-Time Interactive World Model (20+ FPS, Transformer-Only Inference)
Build a procedurally generated interactive environment running entirely on transformer inference at 20+ FPS (no game engine, no pre-rendered assets). Examples: surgical training environment, space station operations simulation, underwater research station, or city navigation. Use Oasis architecture (DiT + ViT with persistent KV cache for frame-to-frame continuity). Fine-tune DiT on domain-specific video dataset for 6-8 hours during hackathon. Judge appeal: validates Etched's Sohu thesis + Prime Intellect's distributed inference + shows Anthropic a qualitatively new interactive experience.

**Demo moment:** Hand keyboard to judge, show FPS counter at 22. Disconnect 4 H100s, FPS drops to 6 (unusable). Reconnect, back to 22. Close: "On Etched's Sohu, this runs at 4K with 100B parameters."

---

### Pattern 2: Serial-Refinement Agent with Process Reward Model Scoring
Agent system applying PRM scoring to 8 sequential refinement steps, proving serial depth beats parallel breadth at fixed compute. Apply to verifiable domain: code review (FrontierCode benchmark), formal math reasoning (MATH dataset, AMC/AIME), or scientific hypothesis evaluation. Use Qwen2.5 Math PRM or fine-tuned domain PRM. Policy model (Llama 70B or Claude) on 6 H100s; PRM on 2 H100s. Benchmark: 8-step serial refinement vs. 64-sample parallel baseline at same token budget vs. GPT-4 baseline.

**Demo moment:** "Diamond" difficulty FrontierCode task. First draft (low quality, PRM score 0.42—below threshold). Agent revises. Step 8: PRM score 0.84, solution passes all evaluators. Baseline GPT-4 failed at step 3. Live graph showing reasoning quality improving over 8 iterations.

---

### Pattern 3: Sub-300ms Voice AI with Visible Latency Dashboard (2x Faster than Mercor's Monty)
Conversational voice AI responding in <300ms (half of Monty's 700ms production latency) applied to high-stakes domain. Tech stack: Local Whisper large-v3 (~20ms), speculative-decoded Llama 70B (~150ms TTFT), Cartesia Sonic streaming TTS (~90ms), Pipecat smart-turn-v3 turn detection, Daily.co WebRTC. Target: 260–350ms total. Apply to emergency dispatch, surgical assistant, trading floor copilot, or incident response.

**Demo moment:** Three-part: (1) ChatGPT Voice responding to question (~1.5s). (2) Your system responding to same question (<300ms). (3) Hand microphone to judge—they ask something unexpected, still works in <350ms. Real-time latency dashboard shows component breakdown. Close: "Monty: 700ms. We: 280ms. Etched Sohu: sub-50ms."

---

## 24-Hour Timeline

**Hours 0–2: Alignment & Planning**
All team members align on project idea, one-sentence insight, demo moment. Write demo script before code. Identify baseline for benchmarking. Agree on success metrics. Verify API keys + model downloads. Download base models to H100s during discussion (don't waste GPU idle time).

**Hours 2–4: Architecture & Tooling Setup**
Set up vLLM (open models), Pipecat (voice), or Oasis fork (world model). Run smoke test—first inference, first API call, first frame. Identify + fix first blocker. Write evaluation harness (measure progress from hour 4 onward).

**Hours 4–10: Core Implementation**
Build core technical contribution: PRM loop, DiT training, speculative decoding, MCTS, or RL training. If training job exists, start it now—let it run autonomously on H100s while you build surrounding system.

**Hours 10–16: Integration & Evaluation**
Connect core system to evaluation framework. Run baseline + system measurements. Note delta (this becomes benchmark claim). Fix integration bugs. Start demo UI—functional by hour 16.

**Hours 16–20: Polish & Hardening**
Polish demo UI—add visible latency counter, FPS counter, benchmark number. Rehearse golden path demo 3 times. Build "judges go off-script" scenario. Prepare one-sentence insight + pitch opening (problem statement, named user).

**Hours 20–22: Buffer**
Expect one unexpected failure. Fix without panic.

**Hours 22–24: Freeze & Rehearse**
No new features. Full demo rehearsal with team. Practice interactive handoff. Prepare backup recording. Prepare 20-second verbal summary: "Our system demonstrates that serial refinement with PRM scoring achieves 47% on FrontierCode Diamond vs. 13% GPT-4 baseline, using sequential inference-time compute rather than parallel sampling."

---

## Prize Details & Logistics

**Prize Pool:** $100K+ total across all three iterations. First place: $50K (confirmed from LLaDA-R1 iteration 1). Total iteration 1: $60K ($40K first place).

**Compute Provided:** 8x NVIDIA H100s per team (provided by Prime Intellect at iteration 3). Allocation is NVIDIA, not Etched Sohu; Etched is judge/co-host, not compute provider.

**Location & Time:** San Francisco. 24-hour format.

**Tracks:** (1) Agents, (2) Real-Time & Interactive, (3) Talent Marketplace + Applied AI.

---

## Past Winners & Publications

**Iteration 1 Winner (Feb 28 – Mar 1, 2025):**
- **LLaDA-R1** (Stanford team, Zeyneb N. Kaya): First application of RL to adaptive inference-time compute for diffusion language models. Won $40K. Later co-founded Topological (YC S25). Evolved into ICLR 2026 publication.

**Iteration 1 Top-6 Finalists:**
- **Backmasking:** Process Reward Model scoring for diffusion LMs. ~2x improvement on GSM8K vs. baseline. Later became ICML 2025 publication.
- **Interruption Tokens:** "34.5% improvement in refusal rate on SimpleQA with only 3.5% accuracy loss." Later became publication.
- Two additional finalists → ICLR 2026 publications.

**Key Insight:** First iteration taught that this hackathon rewards research-grade algorithmic novelty applied to inference-time compute, not integration depth or UI polish. The question is not "can I build an agent?" but "what new insight about inference-time compute can I demonstrate in 24 hours?"

---

## Universal Winning Patterns (From Cross-Hackathon Analysis)

1. **Algorithmic Novelty Over Polished Scaffolding:** Test—"Does your project prove something belonging in a research paper abstract?" If yes, you're in the right territory.

2. **Speed as a UX Feature, Not Backend Metric:** Make inference-time compute visible. Latency counter, FPS counter, tokens-per-second display, interactive compute budget slider. Mercor built Monty around 700ms because it's a meaningful UX threshold.

3. **The Interactive Handoff:** Hand keyboard/microphone to judge and have it work on something they chose. Oasis works because anyone can play it. GibberLink works because anyone can hear robot language.

4. **Verifiable Outcomes With Quantitative Claims:** "19.2% relative improvement over baseline," "34.5% improvement in refusal rate," "~2x improvement on GSM8K." Identify success metric + baseline before building. Run baselines at hour 2, your measurements at hour 20.

5. **The One-Sentence Insight:** Every winning demo has one sentence legible to non-technical judges hiding significant complexity. "The AIs switched to robot language when they realized they were both AI." "We made a diffusion LM decide how hard to think before answering." Write it before building—if you can't write it, you don't have a coherent project.

---

## Anthropic-Specific Hackathon Winners (Built with Claude Code Track History)

**Opus 4.6 1st Place:** CrossBeam (Mike Brown, California attorney)—housing permit parser using parallel sub-agents, zero hand-typed code. $30K API credits.

**Opus 4.6 2nd Place:** Elisa (Jon McBee)—block-based visual IDE with AI code generation.

**Opus 4.6 3rd Place:** PostVisit.ai (Michał Nedoszytko, physician)—healthcare suite explaining diagnoses + surfacing clinical evidence.

**Opus 4.7 Gold:** MedKit (Bedirhan Keskin, Istanbul doctor)—voice-based clinical simulator training medical students.

**Opus 4.7 Silver:** Wrench Board (Alexis Chapellier, French Alps)—AI board-level electronics diagnostics.

**Opus 4.7 Bronze:** Maieutic (Paula Vasquez-Henriquez, Chile CS teacher)—coding platform requiring students to explain reasoning before writing code.

**Pattern:** Domain expertise (doctors, attorneys, teachers) consistently wins over pure coding ability. The ratio of domain expertise to engineering skill in winning submissions is inverted from typical hackathons.

---

## Key References & Citations

- **Kevin-32B (Cognition, May 2025):** Serial refinement beats parallel breadth. 8 sequential steps with environmental feedback > 64 parallel samples at fixed compute.
- **Oasis (Decart + Etched):** Minecraft world generation at 20 FPS, pure neural inference, no game engine. Open-source codebase: github.com/etched-ai/open-oasis.
- **Monty (Mercor):** Production AI interviewer. 10,000 interviews/day. 700ms end-to-end latency. Reference point for Real-Time track.
- **APEX Benchmark (Mercor):** 400 expert-crafted evaluation cases across investment banking, consulting, law, medicine.
- **Lab Platform (Prime Intellect, May 2026):** Hosted post-training loop for self-improving agents. Designed for hackathon use cases.
- **Qwen2.5 Math PRM (HuggingFace):** Available for PRM-based agent refinement.
- **FrontierCode Benchmark:** Cognition's standard—code a maintainer would actually merge, not just tests passing.
- **Pipecat (Open-source):** Framework powering Mercor's Monty. github.com/pipecat-ai/pipecat

