# AI-Based Adaptive Noise Cancellation with DSP Filters

## A Complete Academic Report on the Q-Learning System, Algorithms, and Project Design

**Document type:** Technical / AI coursework project report  
**Focus:** How the artificial intelligence models work — especially tabular Q-learning, Q-table formation, state–action–reward design — in the context of Artificial Intelligence subjects (RL, MDPs, contextual bandits, supervised learning)  
**Project artifact:** `models/q_table.json` (trained with 5,000 episodes)  
**Companion models:** Supervised GBRT filter selector (`models/selector.joblib`), optional Random Forest noise classifier  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Where This Project Sits in the AI Subject](#3-where-this-project-sits-in-the-ai-subject)
4. [System Architecture](#4-system-architecture)
5. [Classical DSP as the Action Space](#5-classical-dsp-as-the-action-space)
6. [Markov Decision Process Formulation](#6-markov-decision-process-formulation)
7. [State Representation: From Audio to Discrete State](#7-state-representation-from-audio-to-discrete-state)
8. [Reward Design: SI-SDR as the Learning Signal](#8-reward-design-si-sdr-as-the-learning-signal)
9. [The Q-Learning Algorithm](#9-the-q-learning-algorithm)
10. [How the Q-Table Is Formed (Step by Step)](#10-how-the-q-table-is-formed-step-by-step)
11. [Anatomy of `q_table.json`](#11-anatomy-of-q_tablejson)
12. [Reading Our Trained Policy](#12-reading-our-trained-policy)
13. [Exploration vs Exploitation (ε-Greedy)](#13-exploration-vs-exploitation-ε-greedy)
14. [Inference: How the Agent Chooses a Filter at Test Time](#14-inference-how-the-agent-chooses-a-filter-at-test-time)
15. [Training Data and Mixture Generation](#15-training-data-and-mixture-generation)
16. [The Supervised Selector (Second AI Path)](#16-the-supervised-selector-second-ai-path)
17. [Continuous Features (35-D) vs Discrete RL States](#17-continuous-features-35-d-vs-discrete-rl-states)
18. [Optional Noise Classifier](#18-optional-noise-classifier)
19. [Evaluation Metrics and Experimental Protocol](#19-evaluation-metrics-and-experimental-protocol)
20. [Hyperparameters and Design Choices Explained](#20-hyperparameters-and-design-choices-explained)
21. [Worked Numerical Example](#21-worked-numerical-example)
22. [Limitations, Failure Modes, and Why Real Audio Is Hard](#22-limitations-failure-modes-and-why-real-audio-is-hard)
23. [How to Improve Accuracy (Conceptual Roadmap)](#23-how-to-improve-accuracy-conceptual-roadmap)
24. [Module Map and Reproducibility](#24-module-map-and-reproducibility)
25. [Glossary of AI Terms Used in This Project](#25-glossary-of-ai-terms-used-in-this-project)
26. [Conclusion](#26-conclusion)
27. [Appendix A — Full Action Catalog](#appendix-a--full-action-catalog)
28. [Appendix B — Q-Table State Catalog (Current Run)](#appendix-b--q-table-state-catalog-current-run)
29. [Appendix C — Suggested Report Figures / Diagrams](#appendix-c--suggested-report-figures--diagrams)

---

## 1. Abstract

This project builds an **adaptive speech enhancement** system in which an artificial intelligence agent selects one classical digital signal processing (DSP) filter for a noisy speech recording. The core AI method studied here is **tabular Q-learning**: a reinforcement learning (RL) algorithm that estimates the expected usefulness of each filter in each acoustic situation.

The agent never invents a new denoiser. Instead, it learns a **policy** — a mapping from a compact description of the noisy clip (the **state**) to one of twelve DSP **actions** (Wiener filtering, spectral subtraction, notch filters for electrical hum, and others). During training, the agent is rewarded when the chosen filter improves speech quality, measured by **scale-invariant signal-to-distortion ratio (SI-SDR)** against a clean reference. Those reward samples are written into a **Q-table**: a dictionary of state keys to vectors of action values. After training, the table is saved as `models/q_table.json` and used at inference without needing the clean reference.

A second, stronger AI path trains a bank of **Gradient Boosted Regression Trees (GBRT)** on 35 continuous acoustic features to predict expected SI-SDR gain for every filter. That supervised selector is recommended for real audio; the Q-learning agent remains the pedagogically clearest RL baseline.

This report explains every AI concept used in the project — MDP, policy, Q-values, ε-greedy exploration, reward shaping, discrete state binning, one-step (contextual) RL, and how the concrete Q-table in this repository was formed — so that reading the report alone is enough to understand what was built and why.

---

## 2. Introduction and Problem Statement

### 2.1 The practical problem

Speech recorded in the real world is almost always contaminated by environmental noise: fan hum, traffic, keyboard clicks, wind, electrical mains interference (50 Hz or 60 Hz), and combinations of these. Classical DSP offers many filters that help in *some* conditions and hurt in others:

- A **50 Hz notch** helps when there is electrical hum, but can damage speech if used blindly.
- **Wiener filtering** helps for stationary broadband noise, but may smear impulsive noise.
- **Median declicking** helps clicks/transients, but is the wrong tool for continuous rumble.

A human engineer can listen and pick a filter. The AI goal of this project is to **automate that choice**.

### 2.2 What we do *not* do

We do not train a deep neural network that outputs a waveform (as in modern speech enhancement models like Demucs or Conv-TasNet). We deliberately keep:

- a **small, interpretable action space** of classical filters;
- a **tabular RL agent** whose knowledge is literally readable as a JSON table;
- a **supervised alternative** that is more accurate but still selects from the same filter bank.

This makes the project suitable for AI coursework: every decision of the agent can be traced to a Q-value.

### 2.3 Research / learning questions answered by the project

1. How can noise cancellation be framed as an **AI decision problem**?
2. What is an appropriate **state**, **action**, and **reward** for filter selection?
3. How does **Q-learning** fill a Q-table from experience?
4. Why does a one-step MDP with γ = 0 behave like a **contextual bandit**?
5. Why does the same agent that looks good on synthetic mixtures often fail on **real** audio?

---

## 3. Where This Project Sits in the AI Subject

### 3.1 Artificial Intelligence → Machine Learning → Reinforcement Learning

In a standard AI syllabus:

| Layer | This project’s instance |
|-------|-------------------------|
| **AI** | An agent that perceives an environment (noisy audio) and acts (chooses a filter) to maximize a performance measure |
| **ML** | Parameters (Q-values or GBRT trees) are learned from data, not hand-coded if–then rules for every noise type |
| **RL** | Feedback is a **scalar reward** after an action; the agent is not given the “correct filter label” up front in the Q-learning path |
| **Supervised learning** | The GBRT selector *is* given labels: measured Δ SI-SDR for every filter on every training clip |

### 3.2 Intelligent agent (Russell & Norvig view)

The RL filter selector is an agent with:

- **Percepts:** features computed from the noisy waveform.
- **Actions:** the twelve DSP filters (plus “do nothing”).
- **Performance measure:** SI-SDR improvement (plus a tiny complexity penalty).
- **Environment:** the audio mixture process (clean + noise + SNR).

### 3.3 Why this is RL, not only “if spectral flatness high then Wiener”

Hand-written rules require an engineer to invent thresholds for every case. Q-learning instead **estimates from trials** how good each action was in each state. After enough episodes, the Q-table *is* the learned rulebook.

### 3.4 Contextual bandit interpretation (important for examiners)

Because each episode has **one decision** and then ends (`γ = 0`, no next-state bootstrap), the problem is equivalent to a **contextual multi-armed bandit**:

- **Context** = acoustic state string (e.g. `0_1_0_0_1`)
- **Arms** = 12 filters
- **Payoff** = Δ SI-SDR (clipped, with small penalty)

Calling it “Q-learning” is still correct: we use the Q-learning update equation. With γ = 0 it reduces to an incremental average of immediate rewards — which is exactly what you want for one-shot filter selection.

### 3.5 Related AI topics touched by the project

- **Feature engineering** (hand-crafted acoustic descriptors)
- **Discretization / quantization** of continuous signals into tabular states
- **Exploration–exploitation dilemma** (ε-greedy)
- **Off-policy style tabular updates** (Q-learning’s max operator; here simplified by γ = 0)
- **Ensemble supervised learning** (gradient boosting)
- **Domain shift / generalization** (synthetic train → real test)

---

## 4. System Architecture

### 4.1 End-to-end pipeline

```text
 Clean speech WAV  ──┐
                     ├── mix_at_snr(SNR) ──► Noisy mixture
 Noise WAV  ─────────┘
                            │
                            ▼
                 Feature extraction
                 ┌──────────┴──────────┐
                 │                     │
         extract_state()        extract_features()
         (5 bins → string)      (35 continuous dims)
                 │                     │
                 ▼                     ▼
         Q-learning agent       GBRT selector bank
         (q_table.json)         (selector.joblib)
                 │                     │
                 └──────────┬──────────┘
                            ▼
                   Chosen action name
                            │
                            ▼
                   apply_filter(action, audio, sr)
                            │
                            ▼
                      Enhanced WAV
```

### 4.2 Training vs inference asymmetry

| Phase | Clean reference available? | What is learned / used |
|-------|----------------------------|-------------------------|
| **Training (Q-learning)** | Yes | Reward = SI-SDR improvement; updates Q(s,a) |
| **Training (supervised)** | Yes | Labels = SI-SDR improvement for *every* filter via full sweep |
| **Inference (deployed)** | No | Only noisy audio → state/features → policy → filter |

This asymmetry is standard in speech enhancement research: clean audio is available when building datasets in a lab, but not when a user uploads a phone recording.

### 4.3 Software entry points

| Task | Command / module |
|------|------------------|
| Generate synthetic data | `python -m noise_suppression.generate_synthetic_samples` |
| Train Q-agent | `python -m noise_suppression.train_agent --episodes 5000` |
| Enhance with Q-policy | `python -m noise_suppression.enhance --method auto` |
| Enhance with supervised selector | `python -m noise_suppression.enhance --method selector` |
| Gradio UI | `python -m noise_suppression.app` |
| Benchmark | `python -m noise_suppression.evaluate` |

---

## 5. Classical DSP as the Action Space

### 5.1 Design philosophy

The AI does not replace DSP; it **schedules** DSP. Each action is a complete, deterministic filter implemented in `filters.py`. The learning problem is therefore finite and discrete: choose among 12 named treatments.

### 5.2 The twelve actions (order = Q-vector index)

| Index | Action name | Intended noise situation |
|------:|-------------|--------------------------|
| 0 | `keep_original` | Already clean enough; any filter risks distortion |
| 1 | `spectral_subtraction` | Additive noise with estimable noise spectrum |
| 2 | `wiener` | Stationary noise; balanced Wiener gain |
| 3 | `wiener_mild` | Mild noise; avoid over-suppression |
| 4 | `wiener_aggressive` | Strong stationary noise |
| 5 | `speech_bandpass` | Out-of-band rumble / ultrasonic hiss |
| 6 | `notch_50hz` | Mains hum (many countries) |
| 7 | `notch_60hz` | Mains hum (Americas etc.) |
| 8 | `highpass_80hz` | Low-frequency rumble / proximity boom |
| 9 | `log_mmse` | Classical statistical speech enhancement |
| 10 | `median_declicker` | Impulsive clicks, typing, claps |
| 11 | `bandpass_then_wiener` | Combined band-limit + denoise |

### 5.3 Why including `keep_original` matters

In AI terms, the null action is critical. Without it, the agent is forced to “always intervene,” which can *reduce* SI-SDR. The reward also subtracts a tiny **0.01** penalty for any non-identity action so that, when two actions are nearly equal, the agent prefers doing nothing — a form of **Occam’s razor / complexity regularization** in the reward.

---

## 6. Markov Decision Process Formulation

### 6.1 Formal tuple

An MDP is \( (S, A, P, R, \gamma) \). In this project:

| Symbol | Meaning in this project |
|--------|-------------------------|
| \(S\) | Discrete acoustic states (strings of 5 bins); at most 72 theoretically |
| \(A\) | The 12 filter names / indices |
| \(P(s'|s,a)\) | Not modeled explicitly; each episode is **terminal** after one action |
| \(R(s,a)\) | Δ SI-SDR (clipped), with −0.01 if \(a \neq\) keep_original |
| \(\gamma\) | **0.0** (no future discounting) |

### 6.2 Episode definition

One training **episode** =

1. Sample one clean file and one noise file.
2. Mix at a random SNR from \(\{-5, 0, 5, 10, 15\}\) dB.
3. Observe state \(s = \texttt{extract\_state}(\text{noisy})\).
4. Choose action \(a\) with ε-greedy.
5. Apply filter → enhanced audio.
6. Compute reward \(r\).
7. Update \(Q(s,a)\).
8. **Stop** (no second decision in that episode).

### 6.3 Why γ = 0 is intentional, not a mistake

If the agent selected filters **frame-by-frame** over time, future rewards would matter and γ > 0 would be appropriate. Here the decision is **once per clip**. There is no meaningful next state for the same decision problem, so γ = 0 is the theoretically correct simplification. The update becomes:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big( r - Q(s,a) \big)
\]

which is incremental averaging of observed rewards for that (state, action) pair.

### 6.4 Policy

The **greedy policy** is:

\[
\pi(s) = \arg\max_{a \in A} Q(s,a)
\]

with random tie-breaking among equal maxima during training exploration; at inference, `numpy.argmax` (first max) is used after safeguards for unseen states.

---

## 7. State Representation: From Audio to Discrete State

### 7.1 Why discretize?

Tabular Q-learning needs a **finite key**. Continuous MFCC vectors cannot be used directly as dictionary keys without function approximation (neural Q-networks). The project therefore compresses the noisy clip into **five coarse bins**, concatenated into a string:

```text
"{flatness}_{flux}_{low_band}_{high_band}_{variability}"
```

Example: `0_1_0_0_1`

### 7.2 The five acoustic features (before binning)

All features are computed from a short-time Fourier transform (STFT) of the **noisy** signal only (Hann window, ~32 ms frames, 75% overlap).

#### (1) Spectral flatness (tonal vs noise-like)

For each frame, geometric mean of power divided by arithmetic mean. High flatness ≈ noise-like spectrum; low flatness ≈ tonal / harmonic (speech or hum).

**Bins:**  
- `0` if flatness < 0.08 (more tonal)  
- `1` if < 0.25  
- `2` otherwise (very noise-like)

#### (2) Spectral flux (how much the spectrum changes over time)

Mean L2 distance between consecutive **normalized** magnitude frames. High flux ≈ nonstationary / speech-like dynamics or impulsive events.

**Bins:**  
- `0` if flux < 0.035  
- `1` otherwise

#### (3) Low-band power share (< 250 Hz)

Fraction of power below 250 Hz. High → rumble, engine, proximity boom.

**Bins:**  
- `0` if share < 0.20  
- `1` otherwise

#### (4) High-band power share (≥ 4000 Hz)

Fraction of power at/above 4 kHz. High → hiss, wind air turbulence, some impulsive spectra.

**Bins:**  
- `0` if share < 0.05  
- `1` if < 0.15  
- `2` otherwise

#### (5) Frame-energy variability

Coefficient of variation of frame RMS: \(\mathrm{std}/\mathrm{mean}\). High → speech-like amplitude modulation or bursts.

**Bins:**  
- `0` if variability < 0.40  
- `1` otherwise

### 7.3 Theoretical state-space size

\[
|S|_{\max} = 3 \times 2 \times 2 \times 3 \times 2 = 72
\]

In the trained artifact after 5,000 episodes on synthetic data, only **7 states** appear. That sparsity is itself an important AI lesson: the agent only learns what it visits.

### 7.4 Short-clip edge case

If the waveform is shorter than 32 samples, the code returns a fixed state `"1_0_0_0_0"` so training does not crash.

### 7.5 Conceptual meaning of a state

A state is a **coarse fingerprint** of the noise-and-speech mixture:

- “Does this clip look rumble-heavy?”
- “Is the spectrum flat like white noise?”
- “Is energy jumping frame to frame like speech or clicks?”

It is *not* a semantic label like “keyboard” or “traffic.” That semantic gap is one reason real-world accuracy is limited.

---

## 8. Reward Design: SI-SDR as the Learning Signal

### 8.1 Why not “accuracy” or “correct filter class”?

In supervised classification you need labels like “this clip should use Wiener.” Creating those labels by hand is expensive. Instead, RL uses a **scalar quality score** that can be computed automatically whenever clean audio is available.

### 8.2 SI-SDR definition (scale-invariant SDR)

Let \(s\) be clean speech and \(\hat{s}\) the estimate (noisy or enhanced). After mean-centering:

1. Project \(\hat{s}\) onto \(s\):
   \[
   s_{\mathrm{target}} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2}\, s
   \]
2. Error: \(e = \hat{s} - s_{\mathrm{target}}\)
3. Score:
   \[
   \mathrm{SI\text{-}SDR} = 10 \log_{10} \frac{\|s_{\mathrm{target}}\|^2}{\|e\|^2}
   \]

Scale-invariance means overall loudness differences do not dominate the metric — important because filters may change level, and the project also RMS-matches outputs.

### 8.3 Reward used in training

\[
r = \underbrace{\mathrm{SI\text{-}SDR}(clean, enhanced)}_{\text{after filter}}
  - \underbrace{\mathrm{SI\text{-}SDR}(clean, noisy)}_{\text{before filter}}
  - 0.01 \cdot \mathbf{1}[a \neq \texttt{keep\_original}]
\]

Then:

\[
r \leftarrow \mathrm{clip}(r,\ -15,\ +15)
\]

**Interpretation:**

- Positive \(r\) → the filter helped relative to doing nothing (on the SI-SDR axis).
- Negative \(r\) → the filter hurt.
- Clipping prevents extreme outliers from destabilizing updates.
- The 0.01 term is **reward shaping** that breaks ties toward simpler policies.

### 8.4 Other metrics (evaluation, not Q-update)

- **STOI** (Short-Time Objective Intelligibility): correlates with human intelligibility.
- **PESQ-WB** (Perceptual Evaluation of Speech Quality): perceptual MOS-like score (optional dependency).

The Q-agent optimizes SI-SDR improvement only. An action that improves PESQ but not SI-SDR will not be preferred by this reward.

---

## 9. The Q-Learning Algorithm

### 9.1 Classical Q-learning reminder

Watkins’ Q-learning learns an action-value function \(Q(s,a)\) approximating the expected return of taking action \(a\) in state \(s\) and acting optimally thereafter:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)
\]

### 9.2 Our specialized form

With \(\gamma = 0\) and terminal episodes:

\[
Q(s,a) \leftarrow Q(s,a) + 0.15 \big( r - Q(s,a) \big)
\]

Equivalently:

\[
Q(s,a) \leftarrow (1 - 0.15)\, Q(s,a) + 0.15\, r
\]

So each visit blends 85% of the old estimate with 15% of the new reward. This is **exponential moving averaging** of rewards for that cell.

### 9.3 Implementation class

`QLearningFilterSelector` in `agent.py` stores:

```text
q_table: dict[str, list[float]]
```

- Key: state string  
- Value: length-12 list of Q-values aligned with `ACTIONS`

Lazy initialization: unseen states start as `[0,0,...,0]`.

### 9.4 Learning rate α = 0.15

- Too high → noisy, unstable Q-values (each bad sample swings the table).
- Too low → very slow learning; needs more episodes.
- 0.15 is a moderate choice for thousands of episodes on a small state space.

### 9.5 Seed and reproducibility

Training uses `seed=42` for NumPy and Python `random`, so mixture sampling and ε-greedy draws are reproducible for a given dataset.

---

## 10. How the Q-Table Is Formed (Step by Step)

This section is the heart of the report: **exactly how `q_table.json` comes into existence**.

### Step 1 — Prepare experience sources

Place clean speech under `data/clean/` and noise-only files under `data/noise/`.  
Optionally run:

```text
python -m noise_suppression.generate_synthetic_samples
```

which synthesizes formant-based pseudo-speech and several noise types (hum, engine, white noise, wind, clapping, keyboard).

### Step 2 — Initialize an empty agent

```text
Q = {}   # no states yet
α = 0.15
γ = 0.0
```

### Step 3 — Repeat for T episodes (e.g. T = 5000)

For episode \(t = 0,1,\ldots,T-1\):

1. **Sample** clean path \(c\), noise path \(n\).
2. **Load & align** sample rates; truncate clean to ≤ 12 seconds for speed.
3. **Mix** at SNR \( \in \{-5,0,5,10,15\} \) dB → `noisy`.
4. **Observe** \( s = \texttt{extract\_state}(noisy) \).  
   If \(s\) is new, create `Q[s] = [0]*12`.
5. **Explore or exploit:**
   \[
   \varepsilon_t = \max\big(0.05,\ 0.35 \cdot (1 - t/T)\big)
   \]
   With probability \(\varepsilon_t\), pick a random action index; else pick \(\arg\max Q[s]\).
6. **Act:** `enhanced = apply_filter(ACTIONS[a], noisy, sr)`.
7. **Reward:** compute \(r\) as in Section 8; clip to \([-15,15]\).
8. **Learn:**  
   \( Q[s][a] \leftarrow Q[s][a] + 0.15 (r - Q[s][a]) \).
9. Log progress every 10% of episodes (mean of last 100 rewards).

### Step 4 — Persist the table

Serialize to JSON:

- list of action names (for version safety),
- full `q_table` dictionary,
- metadata (episode count, file counts, mean reward, seed, timing).

### Step 5 — What “formed” means scientifically

The Q-table is **not** a neural network weight matrix. It is an empirical estimate:

\[
Q(s,a) \approx \mathbb{E}[r \mid s,a]
\]

estimated by stochastic approximation from the training distribution of mixtures. If that distribution differs from real-user audio, the table’s argmax can be wrong — classic **distribution shift**.

### Step 6 — What our actual run produced

From `models/q_table.json` metadata:

| Field | Value |
|-------|------:|
| Episodes | 5000 |
| Clean files | 5 |
| Noise files | 7 |
| Seed | 42 |
| Mean training reward | ≈ 3.02 dB |
| Training time | ≈ 116 s |
| Distinct states stored | 7 |

Training reward rose from near zero / slightly negative early on toward roughly +3–4 dB mean reward in later checkpoints — evidence that the agent was learning *on that synthetic distribution*.

---

## 11. Anatomy of `q_table.json`

### 11.1 Top-level schema

```json
{
  "format_version": 1,
  "algorithm": "tabular_q_learning_one_step_mdp",
  "actions": [ "...12 names..." ],
  "q_table": {
    "0_1_0_0_1": [ Q0, Q1, ..., Q11 ],
    "...": [ ... ]
  },
  "metadata": { ... }
}
```

### 11.2 Why `actions` is stored

On load, the code checks that the saved action list matches the current `ACTIONS` in `filters.py`. If someone reorders filters in code but reuses an old table, indices would silently point to the wrong DSP routines. The check prevents that class of bug.

### 11.3 Sparse storage

Only visited states exist. The other ~65 theoretical states are absent. At inference, missing keys are handled by a **default policy** (`spectral_subtraction`), not by inventing zeros that would prefer `keep_original` by accident (because index 0 starts at 0.0).

---

## 12. Reading Our Trained Policy

For each stored state, the **greedy action** is the index of the maximum Q-value.

### 12.1 State `0_1_0_0_1`

Q-values (rounded):

| Action | Q ≈ |
|--------|-----:|
| keep_original | 0.00 |
| spectral_subtraction | −7.80 |
| wiener | −1.85 |
| wiener_mild | −1.29 |
| wiener_aggressive | −2.10 |
| speech_bandpass | 0.29 |
| notch_50hz | 0.31 |
| notch_60hz | 0.24 |
| **highpass_80hz** | **+4.08** ← selected |
| log_mmse | −6.92 |
| median_declicker | −10.37 |
| bandpass_then_wiener | −2.04 |

**AI interpretation:** In this bin combination (moderate flux, low high-band share, high energy variability), training experience taught that removing low-frequency content helped most, while aggressive spectral methods often hurt.

### 12.2 State `0_1_1_0_0`

Highest Q ≈ **+5.93** for `bandpass_then_wiener`.  
Also strong: `log_mmse` (~5.04), `speech_bandpass` (~5.02).

**Interpretation:** Low-band energy is high (third digit = 1), so band-limiting plus Wiener was rewarded on average.

### 12.3 State `2_0_0_2_0`

Highest Q ≈ **+2.14** for `bandpass_then_wiener`; `log_mmse` also positive (~1.56).  
This state has high spectral flatness (first digit 2) and high high-band share (fourth digit 2) — more noise-like / hissy — and the policy prefers combined suppression over declicking (declicker Q negative).

### 12.4 Policy as a readable rulebook

Unlike a deep net, you can open the JSON and write English rules:

> IF state is `0_1_0_0_1` THEN use `highpass_80hz`.  
> IF state is `0_1_1_0_0` THEN use `bandpass_then_wiener`.

That interpretability is a major pedagogical advantage of tabular RL.

---

## 13. Exploration vs Exploitation (ε-Greedy)

### 13.1 The dilemma

- **Exploit:** always pick \(\arg\max Q(s,a)\) — may never try a better filter that started with unlucky early rewards.
- **Explore:** try random filters — gathers data but temporarily lowers training reward.

### 13.2 Schedule used

\[
\varepsilon_t = \max\big(0.05,\ 0.35 \cdot (1 - t/T)\big)
\]

| Phase | Approx. ε | Behavior |
|-------|-----------|----------|
| Start (t≈0) | 0.35 | 35% random actions |
| Mid (t≈T/2) | ~0.175 | more greedy |
| End | 0.05 | 5% random forever (never fully freeze exploration) |

Keeping a floor of 0.05 is a common RL practice so late training still occasionally corrects wrong Q estimates.

### 13.3 Tie-breaking during exploration/exploitation

If several actions share the same max Q, the agent samples uniformly among them during `choose_action_index`. This avoids an arbitrary alphabetical bias when many zeros remain early in training.

---

## 14. Inference: How the Agent Chooses a Filter at Test Time

### 14.1 Pipeline at test time

1. Load noisy WAV.
2. Compute `state = extract_state(audio, sr)`.
3. `action = agent.select(state)`.
4. `output = apply_filter(action, audio, sr)`.
5. Write enhanced WAV / play in Gradio.

No clean reference, no reward, no Q-update.

### 14.2 `select()` safeguards

```text
if state not in q_table:
    return "spectral_subtraction"   # unseen context
if all Q-values ≈ equal and max ≤ 0:
    return "spectral_subtraction"   # uninformative row
else:
    return ACTIONS[argmax(Q[state])]
```

**Why not `keep_original` for unseen states?**  
Uninitialized or empty knowledge with all zeros would make index 0 (`keep_original`) win by default, even when the clip is badly corrupted. The designers chose a **mild default denoiser** instead — a form of risk-aware fallback policy.

### 14.3 Gradio path

In the UI, choosing **Q-learning Agent** loads `models/q_table.json` (resolved relative to the project root) and runs the same `select()` logic.

---

## 15. Training Data and Mixture Generation

### 15.1 Synthetic generator (quick start)

`generate_synthetic_samples.py` creates:

- **Clean:** multi-syllable formant synthesis (glottal pulse + vowel formants; male/female/child pitches).
- **Noise:** 50/60 Hz hum + harmonics, engine rumble, white noise, wind, impulsive clap/keyboard patterns.
- **Eval splits:** separate `eval_clean` / `eval_noise` folders.

Sample rate: **16 kHz** (`INTERNAL_SR`).

### 15.2 Mixing function `mix_at_snr`

Noise is scaled so that the mixture approximately hits a target signal-to-noise ratio in dB, then added to clean speech. This creates a controlled curriculum of difficulty: −5 dB is harsh; +15 dB is mild.

### 15.3 Optional real corpora path

The repository also supports LibriSpeech (speech) + ESC-50 (environmental noise) with **zero-leakage** splits (speakers and ESC-50 folds do not cross train/val/test). That path feeds the supervised selector more strongly; the Q-agent *can* be trained on the same folders if populated under `data/clean` and `data/noise`.

### 15.4 Dual-noise mixtures (supervised pipeline)

A fraction of supervised mixtures combine two noise sources (`DUAL_NOISE_FRACTION = 0.10`) to better mimic real environments where fan + keyboard may co-occur.

---

## 16. The Supervised Selector (Second AI Path)

### 16.1 Why a second model exists

Tabular Q-learning with 5 bins is easy to explain but limited in accuracy. The project therefore includes a **supervised multi-regressor selector**:

1. Build a manifest of mixtures.
2. Preprocess audio.
3. **Sweep every filter on every clip** → matrix of true Δ SI-SDR labels (full information).
4. Train one `GradientBoostingRegressor` per action to predict that gain from 35 features.
5. At test time: predict all 12 gains → pick argmax (unless best gain < 0.05 dB → `keep_original`).

### 16.2 Algorithm name in AI terms

- **Base learner:** Gradient Boosted Decision Trees (ensemble of shallow trees, additive boosting).
- **Problem type:** Multi-output regression via independent models (one-vs-rest style for actions).
- **Decision rule:** Greedy w.r.t. predicted gain + safety threshold.

### 16.3 Default GBRT hyperparameters

| Hyperparameter | Value |
|----------------|------:|
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| min_samples_leaf | 10 |
| random_state | 42 |

### 16.4 Comparison to Q-learning (for viva / exams)

| Aspect | Q-learning agent | Supervised GBRT selector |
|--------|------------------|---------------------------|
| Feedback | Samples one action per episode | Sees rewards for *all* actions per clip |
| State | 5-bin string | 35 continuous features |
| Knowledge form | Sparse Q-table | Forest of trees |
| Interpretability | Very high (read JSON) | Medium (feature importances) |
| Typical accuracy | Baseline | Recommended |
| Needs clean audio at train | Yes | Yes |
| Needs clean audio at test | No | No |

Full-information supervised training is usually more sample-efficient than bandit-style Q-learning for this problem, because each training clip teaches all 12 arms, not one.

---

## 17. Continuous Features (35-D) vs Discrete RL States

### 17.1 Feature families in `extract_features`

1. **Level / spectrum shape:** RMS dBFS, spectral flatness, flux, entropy, centroid, rolloff  
2. **Band energies:** low/high shares and absolute log band powers  
3. **MFCCs:** 13 mean coefficients (speech timbre / noise texture)  
4. **Noise character:** stationarity, kurtosis, crest factor (impulsiveness)  
5. **Hum detectors:** 50 Hz and 60 Hz power ratios  
6. **SNR proxy:** VAD-based estimate  
7. **Extras:** zero-crossing rate, harmonic-to-noise ratio  

`FEATURE_VERSION = "v2"` versions the vector so old models are not silently mismatched.

### 17.2 Pedagogical contrast

- Discrete RL state = **lossy compression** for tabular methods.  
- Continuous features = **rich representation** for function approximators.

The same audio can be described both ways; the project intentionally shows both AI paradigms on one filter bank.

---

## 18. Optional Noise Classifier

A `RandomForestClassifier` (200 trees, max depth 12) can map the 35 features to ESC-50-style noise categories. In the current main enhance path it is **optional / diagnostic**, not required for filter selection. Conceptually it is **supervised classification**, separate from RL.

---

## 19. Evaluation Metrics and Experimental Protocol

### 19.1 What “good” means

For each method (fixed filter, Q-agent, selector, oracle):

\[
\Delta\mathrm{SI\text{-}SDR},\quad \Delta\mathrm{STOI},\quad \Delta\mathrm{PESQ}
\]

Oracle = per-clip best fixed filter (upper bound if you could cheat and try all filters with a clean reference).

### 19.2 Statistical reporting

Evaluation can report **95% bootstrap confidence intervals** on mean Δ SI-SDR (`n_boot = 1000`), which is good scientific practice for small test sets.

### 19.3 Held-out data discipline

True evaluation must use clean/noise files **not** seen in training. Folders `eval_clean` / `eval_noise` exist for that purpose. Training on the same five synthetic speakers and then testing similar synthetics can look optimistic relative to phone recordings.

---

## 20. Hyperparameters and Design Choices Explained

| Choice | Value | Why it matters |
|--------|-------|----------------|
| α | 0.15 | Step size of Q updates |
| γ | 0.0 | One-step / contextual problem |
| ε start / floor | 0.35 / 0.05 | Exploration schedule |
| Episodes | 2500 default; 5000 used | More visits → stabler Q estimates |
| SNR set | −5…15 dB | Difficulty diversity |
| Max clip | 12 s | Training speed |
| Reward clip | ±15 dB | Robustness to outliers |
| Non-identity penalty | 0.01 | Prefer simpler action on ties |
| Safety threshold (supervised) | 0.05 dB | Avoid tiny predicted gains |
| Internal SR | 16 kHz | Speech processing standard |

Every hyperparameter is an AI design decision: change them and the Q-table changes.

---

## 21. Worked Numerical Example

Suppose at some episode:

- State \(s =\) `0_1_0_0_1`
- Current \(Q(s,\texttt{highpass\_80hz}) = 3.90\)
- Agent selects highpass (exploit)
- Measured \(r = 4.80\) dB

Update:

\[
Q \leftarrow 3.90 + 0.15(4.80 - 3.90) = 3.90 + 0.15\times 0.90 = 3.90 + 0.135 = 4.035
\]

If instead a bad action `median_declicker` scored \(r = -8\):

\[
Q \leftarrow 0 + 0.15(-8 - 0) = -1.2
\]

After many such visits, declicker’s Q becomes largely negative in that state (as in our saved table: ≈ −10.4), so the greedy policy avoids it.

---

## 22. Limitations, Failure Modes, and Why Real Audio Is Hard

### 22.1 Coarse state → collisions

Many different real noises (traffic vs café vs AC) may collapse to the **same** 5-bin string. The Q-table then stores a **compromise** average reward that may not match any of them well.

### 22.2 Unseen states

Real audio may produce state strings never visited in synthetic training → default `spectral_subtraction`, which may be wrong.

### 22.3 Clip-level policy

Noise that starts as hum and later becomes typing cannot get two filters; the agent picks one for the whole file.

### 22.4 Domain shift

Synthetic formant speech ≠ LibriSpeech ≠ smartphone speech. Synthetic hum ≠ real fluorescent ballast hum. RL policies are only as good as the **training distribution**.

### 22.5 Reward–perception mismatch

SI-SDR is objective and convenient, but humans may prefer a filter that scores slightly worse on SI-SDR yet sounds clearer (PESQ/STOI disagreement).

### 22.6 Sparse visits

With only 5 clean × 7 noise files, even 5,000 episodes resample a small world. Q-values can be confident-looking but overfit to that mini-universe.

---

## 23. How to Improve Accuracy (Conceptual Roadmap)

*(Guidance only — not an implementation checklist for this document’s scope.)*

1. **Train on real speech and real noise** (LibriSpeech + ESC-50 or your own recordings).  
2. **Enrich or abandon coarse bins** — finer discretization, or continuous-state methods (DQN, linear FA).  
3. **Prefer the supervised selector** for deployment-quality results.  
4. **Segment-wise decisions** for nonstationary noise.  
5. **More diverse SNRs, dual-noise mixes, longer clips.**  
6. **Evaluate on held-out real recordings** with bootstrap CIs, not only training reward curves.

---

## 24. Module Map and Reproducibility

| Module | AI / system role |
|--------|------------------|
| `agent.py` | Tabular Q-learning class |
| `train_agent.py` | Episode loop, reward, save JSON |
| `features.py` | `extract_state` + `extract_features` |
| `filters.py` | Action space / DSP bank |
| `audio_utils.py` | Mix, SI-SDR, I/O |
| `selector.py` / `train_selector.py` | GBRT path |
| `sweep_filters.py` | Full-information labels |
| `enhance.py` | Inference CLI |
| `evaluate.py` | Benchmarking |
| `app.py` | Gradio interface |
| `generate_synthetic_samples.py` | Synthetic world for quick RL |
| `models/q_table.json` | Learned RL policy artifact |

**Minimum reproduction of the Q-table:**

```text
cd <project-root>
python -m noise_suppression.generate_synthetic_samples
python -m noise_suppression.train_agent --episodes 5000 --output models/q_table.json
```

---

## 25. Glossary of AI Terms Used in This Project

| Term | Meaning here |
|------|----------------|
| **Agent** | Code that chooses a DSP filter |
| **Environment** | Process that produces noisy audio and rewards |
| **State** | Discrete acoustic fingerprint string |
| **Action** | One DSP filter |
| **Reward** | Δ SI-SDR (shaped, clipped) |
| **Episode** | One mixture → one decision → one update |
| **Policy** | Mapping state → action (greedy over Q) |
| **Q-value** | Estimated expected reward for (state, action) |
| **Q-table** | Dictionary storing all Q-values |
| **Exploration** | Trying random filters (ε) |
| **Exploitation** | Using current best Q-values |
| **MDP** | Formal RL environment model |
| **Contextual bandit** | One-step decision with context (= our γ=0 case) |
| **Feature engineering** | Hand-designed audio descriptors |
| **Supervised learning** | Learn from labeled Δ SI-SDR for all filters |
| **Generalization** | Performance on unseen / real audio |
| **Domain shift** | Train distribution ≠ test distribution |

---

## 26. Conclusion

This project demonstrates a complete, explainable AI pipeline for **adaptive DSP filter selection**:

1. Noise cancellation is cast as a **decision problem** in Artificial Intelligence.  
2. The decision problem is formalized as a **one-step MDP / contextual bandit**.  
3. **Tabular Q-learning** estimates \(Q(s,a)\) using SI-SDR improvement as reward.  
4. The **Q-table** is formed by thousands of mix–act–measure–update cycles and saved as JSON.  
5. At inference, the agent reads acoustic state from noisy audio alone and executes the greedy filter.  
6. A **supervised GBRT selector** on 35 features provides a stronger practical alternative while sharing the same DSP action space.  
7. Limitations — coarse states, synthetic training, clip-level decisions — explain why real audio often needs better data and richer models.

If an examiner asks *“How does your AI model work?”*, the short answer is:

> We learn a table of expected speech-quality improvements for each acoustic situation and each classical filter using Q-learning; then we pick the filter with the highest Q-value for the observed situation.

If they ask *“How is the Q-table formed?”*, the short answer is:

> By repeatedly mixing clean speech with noise, observing a discrete state, trying a filter (ε-greedy), scoring the SI-SDR gain, and nudging that state’s action value toward the observed reward with learning rate 0.15, until the JSON table stabilizes.

---

## Appendix A — Full Action Catalog

See Section 5.2 for the indexed list. Implementation details live in `src/noise_suppression/filters.py` (spectral subtraction with noise floor, Wiener variants with strength parameters, Butterworth bandpass/highpass, cascaded IIR notches, Ephraim–Malah-style log-MMSE, median declicker, and the cascade `bandpass_then_wiener`).

---

## Appendix B — Q-Table State Catalog (Current Run)

States present after 5,000-episode synthetic training:

| State key | Greedy action (from argmax Q) | Rough meaning of bins |
|-----------|-------------------------------|------------------------|
| `0_1_0_0_1` | `highpass_80hz` | Low flatness, high flux, not low-heavy, low high-band, high variability |
| `0_1_1_0_0` | `bandpass_then_wiener` | Similar but **low-band heavy** |
| `0_0_0_0_1` | (inspect Q; mild/near-zero row) | Low flux, high variability |
| `1_0_0_0_1` | near-zero competition | Mid flatness |
| `0_1_0_0_0` | `wiener_aggressive` (highest among small positives) | High flux, low variability |
| `1_0_0_1_0` | `wiener` | Mid flatness, mid high-band |
| `2_0_0_2_0` | `bandpass_then_wiener` | Very flat + high high-band (noise-like/hissy) |

*(Exact argmax for near-tied rows should be verified from the JSON when writing results tables for submission.)*

---

## Appendix C — Suggested Report Figures / Diagrams

When converting this Markdown to a printed ~coursework PDF, include:

1. **System block diagram** (Section 4.1).  
2. **MDP diagram:** state → action → reward → terminal.  
3. **ε decay curve** over episodes.  
4. **Q-update flowchart.**  
5. **State binning diagram** (5 features → string).  
6. **Bar chart of Q-values** for one example state (e.g. `0_1_0_0_1`).  
7. **Training reward curve** (episode checkpoint means).  
8. **Comparison table** Q-agent vs supervised selector vs oracle.  
9. **Failure case:** spectrogram of real audio misclassified into a synthetic-trained state.

---

*End of report.*

**Primary artifacts referenced:**  
`src/noise_suppression/agent.py`, `train_agent.py`, `features.py`, `filters.py`, `selector.py`, `audio_utils.py`, `models/q_table.json`, project `README.md`.
