---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
description-zh: # MDP Specification & Formulation Risk Checklist

## Step 1: Extract the MDP Tuple ⟨S, A, T, R, γ⟩

Given any task description, define each component:

### **State Space (S)**
```
- What observations does the agent receive?
- Are states fully observable (MDP) or partially observable (POMDP)?
- Continuous or discrete?
- What features are included / excluded?
- Include: agent state, environment state, goal/task parameters
```

### **Action Space (A)**
```
- Discrete (finite set) or Continuous (ℝⁿ)?
- Low-level (torques, velocities) vs. high-level (subgoals, options)?
- Hierarchical actions needed?
- Constraints on actions (e.g., joint limits, budgets)?
```

### **Transition Function (T)**
```
- Known (model-based) or learned (model-free)?
- Stochastic or deterministic?
- Stationary or non-stationary?
- Physics-based simulator or real-world?
```
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
