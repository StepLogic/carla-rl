# CORL 2026 Paper Strengthening Design

## Context
Paper: "Sparse Topological Map for Navigation using Policy Uncertainty" (Paper-ID [807])
Authors: Elvis Gyaase, Masoud Ataei, Vikas Dhiman
Target: CORL 2026 (8-page limit + unlimited appendix)
Current state: Previously submitted to CORL 2025 (rejected), now targeting resubmission with significant improvements
Existing review feedback: 4 reviewers with concerns about mathematical rigor, evaluation scale, baseline fairness, and ablations

## Paper's Core Claim
Policy uncertainty (entropy of the low-level policy) identifies decision states (junctions) and can be used to build sparse topological maps that achieve comparable navigation performance with 10x fewer nodes than dense baselines.

## Design: Three Pillars

### Pillar 1: Mathematical Framework (Pages 3-4 of paper, full proofs in appendix)

#### 1.1 Definition: Decision State (NEW)
A state s ∈ S is a **decision state** with respect to policy π if:
  H(π(·|s)) > H̄_m + k · MAD_m

where H̄_m is the running median of the entropy over a window of size m, MAD_m is the median absolute deviation, and k is a sensitivity parameter.

This formalizes the informal "locations where the low-level policy is uncertain" and makes the algorithm testable.

#### 1.2 Proposition: Reachability Preservation (NEW)
**Statement**: Under assumptions that (A1) the exploration policy is ergodic over the reachable state space S_R ⊆ S, and (A2) the graph connects sequentially detected decision states along each trajectory, the sparse topological map M = G(V, E) preserves reachability between any two decision states that are connected by a path in S_R.

**Proof sketch**: By construction, edges connect consecutively detected decision states. By A1, each junction is visited with positive probability. By A2, the graph preserves the connectivity of the exploration trajectory. Therefore, if two decision states are reachable in S_R, they are reachable in M.

**Full proof**: In appendix.

#### 1.3 Proposition: Grid-World Correctness (UPGRADED from illustration)
**Statement**: Under the lane-following policy π_l defined in Eq. 1-2 with λ_lane = λ_coll = 5:
- (a) At non-junction states, H(π_l(·|s)) ≤ log(3) · exp(-5) ≈ 0 (only forward has significant probability)
- (b) At T-junctions, H(π_l(·|s)) ≥ log(2) · (1 - 2·exp(-5)) (two actions have comparable probability)
- (c) The entropy-based detection algorithm with k ≥ 2 correctly identifies all junction states

**Full proof**: In appendix.

#### 1.4 Information-Theoretic Bound (NEW)
**Theorem**: For a navigation task with success rate ε over graph M = G(V, E), the minimum number of decision states satisfies:
  |V*| ≥ |D| ≥ I(S; A) / max_{s ∈ D} H(π(·|s))

where D is the set of decision states, I(S; A) is the mutual information between states and actions, and H(π(·|s)) is the policy entropy at state s.

This provides a theoretical justification: fewer nodes are needed when the policy is more certain at most states (the intuition behind our approach).

**Full proof**: In appendix.

#### 1.5 SAC Entropy Connection (NEW)
Formalize the connection between unsquashed entropy and decision-state detection:
- In SAC, the policy is a squashed Gaussian: a = tanh(μ + σ·ε), ε ~ N(0,I)
- The squashed entropy H_squashed is intractable to compute
- The unsquashed entropy H(π_φ) = k/2·log(2πe) + 1/2·log det(Σ_π) is a valid proxy
- Justification: For bounded actions, the Fisher information of the squashed distribution is related to the unsquashed distribution via the change-of-variables formula. The unsquashed entropy monotonically relates to the "decision complexity" at a state.

### Pillar 2: Experiments (Pages 5-7 of paper)

#### Experiment 1: Scaled Evaluation (CRITICAL - addresses reviewer concern R6Ekq.I.4, RWFMW)
- **What**: Evaluate all methods on 50+ trajectories in Town02
- **Why**: Current results have only 10 trajectories (results in multiples of 10%), no error bars
- **Metrics**: SR, SPL, Map Size, Computation Time
- **Statistical test**: Wilcoxon signed-rank test for pairwise comparisons
- **Report**: mean ± std over 5 random seeds for stochastic methods

#### Experiment 2: Generalization to New Towns (addresses R6Ekq.I.4)
- **What**: Evaluate zero-shot transfer to Town01, Town03, Town05
- **Why**: Reviewers asked about generalization
- **Note**: Low-level policy is not retrained, only the map is rebuilt
- **Report**: Per-town SR and SPL

#### Experiment 3: Ablation — Node Placement Strategies (addresses RWFMW.I.5)
- **What**: Compare entropy-based node placement against:
  - (a) Ours (entropy-based, MAD spike detection)
  - (b) Random placement (same number of nodes as ours)
  - (c) Uniform spacing (same number of nodes as ours)
  - (d) Junction oracle (ground truth junction detection)
  - (e) Dense map (all frames, no sparsification)
  - (f) Distance threshold (learned distance function, as in GNM/ViNT)
- **Why**: Proves entropy-based placement is principled, not just a heuristic
- **Metrics**: SR, SPL vs number of nodes (Pareto curve)

#### Experiment 4: Ablation — Spike Detection Parameters (addresses RgqDj.I.4, RWFMW.I.4)
- **What**: Vary W (window), k (threshold), L (lockout) and measure:
  - (a) Spike detection accuracy at junctions (current metric)
  - (b) Navigation SR/SPL for each configuration
  - (c) Number of nodes in the resulting map
- **Why**: Current ablation only measures detection accuracy, not downstream performance

#### Experiment 5: Low-Level Policy Sensitivity
- **What**: Test our mapping framework with different low-level policies:
  - (a) DrQ (SAC-trained, current default)
  - (b) NoMaD as low-level (zero-shot, entropy from diffusion model)
  - (c) GNM as low-level (zero-shot, entropy from distance prediction)
- **Why**: Shows the framework generalizes beyond one specific policy

#### Experiment 6: Map Size vs Performance Pareto Curve (KEY FIGURE)
- **What**: Plot SR and SPL vs number of nodes for all methods
- **Why**: This is the paper's central claim visualization
- **Format**: Line plot with error bars, methods color-coded

### Pillar 3: Paper Structure for CORL 2026

#### Paper Body (8 pages)
1. **Introduction** (1.5 pages): Motivating example, core insight, contributions
2. **Related Work** (0.5 pages): Semi-parametric policies, topological memories, sparse representations
3. **Problem Formulation** (1 page): Formal definitions, reachability proposition (proof in appendix)
4. **Method** (2 pages): Algorithm, entropy detection, navigation, grid-world illustration
5. **Experiments** (2 pages): Scaled evaluation, ablations, generalization
6. **Conclusion** (0.5 pages): Summary, limitations, future work

#### Appendix (unlimited)
- Full proofs of Propositions 1, 2 and Theorem 1
- Detailed experimental setup
- Full ablation tables
- Additional qualitative results (trajectory visualizations)
- Computational efficiency details
- Grid-world detailed results

#### Key Figures
1. **Overview figure** (Fig 1): Current figure, cleaned up
2. **Entropy visualization at junctions** (Fig 2): Current entropy_vz_paper, improved
3. **Map size vs performance Pareto curve** (Fig 3): NEW — central result
4. **Qualitative trajectories** (Fig 4): Current trajectory comparison
5. **Grid-world illustration** (Fig 5): Current, with formal proposition reference

#### Key Tables
1. **Table 1**: Scaled evaluation (50+ trajectories, all towns, with error bars)
2. **Table 2**: Ablation — node placement strategies
3. **Table 3**: Spike detection parameters → navigation performance

## Code Architecture

### Directory Structure
```
corl-2026/
├── paper-sub/
│   ├── main.tex              # CORL 2026 format paper
│   ├── appendix.tex          # Full proofs
│   ├── main.bib              # Bibliography
│   ├── figures/              # Generated figures
│   ├── next-ideas.tex        # Future directions
│   ├── run_all_experiments.sh # Single script to generate all results
│   └── analysis/
│       ├── generate_tables.py # Generate LaTeX tables from results
│       ├── generate_figures.py # Generate all figures
│       └── statistical_tests.py # Wilcoxon tests, confidence intervals
├── experiments/
│   ├── experiment1_scaled_eval.py
│   ├── experiment2_generalization.py
│   ├── experiment3_node_placement_ablation.py
│   ├── experiment4_spike_detection_ablation.py
│   ├── experiment5_policy_sensitivity.py
│   └── experiment6_pareto_curve.py
└── results/
    ├── scaled_eval/
    ├── generalization/
    ├── ablations/
    ├── policy_sensitivity/
    └── pareto/
```

### Experiment Runner Script (run_all_experiments.sh)
A single bash script that:
1. Starts CARLA server (if not running)
2. Runs each experiment sequentially
3. Saves results to structured directories
4. Generates tables and figures
5. Compiles the paper

Can be run with: `bash corl-2026/paper-sub/run_all_experiments.sh --carla-port 2000 --gpu 0`

## Implementation Priority (Given abstract deadline ~May/June)

### Phase 1: Critical Experiments (Week 1-2)
- [ ] Scale evaluation to 50+ trajectories in Town02
- [ ] Add statistical tests and error bars
- [ ] Node placement ablation (random, uniform, oracle, distance-threshold)
- [ ] Spike detection → navigation performance ablation

### Phase 2: Mathematical Framework (Week 2-3)
- [ ] Write formal definitions (decision state)
- [ ] Write and prove Proposition 1 (reachability)
- [ ] Write and prove Proposition 2 (grid-world)
- [ ] Write Theorem 1 (information bound) with proof
- [ ] Write SAC entropy connection section

### Phase 3: Generalization & Additional Experiments (Week 3-4)
- [ ] Zero-shot transfer experiments to Town01, Town03, Town05
- [ ] Low-level policy sensitivity (NoMaD/GNM as low-level)
- [ ] Pareto curve (map size vs performance)
- [ ] Generate all figures

### Phase 4: Paper Writing (Week 4-5)
- [ ] Rewrite paper in CORL 2026 format
- [ ] Write appendix with full proofs
- [ ] Write next-ideas.tex
- [ ] Final review and submission

## Risk Assessment

1. **CARLA instability**: CARLA servers can crash. Mitigation: checkpoint-based experiment design, automatic restart
2. **Compute bottleneck**: Single GPU workstation limits parallelism. Mitigation: sequential experiments, save checkpoints
3. **Baseline fine-tuning**: Fine-tuning baselines on CARLA takes time. Mitigation: start with zero-shot, fine-tune in parallel
4. **Mathematical proof gaps**: Some propositions may need assumptions. Mitigation: state assumptions clearly, prove what we can