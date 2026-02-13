# Semester Project Proposal: Autonomous Agents

---

## 1. Problem Statement
In high-performance autonomous racing, agents typically optimize a static objective (e.g., minimum lap time). However, dynamic race scenarios require adaptable strategies triggered by high-level instructions (e.g., "Defend the inside line," "Push for overtake," or "Conserve tires").

I propose to develop a **Language-Conditioned Racing Agent** that maps natural language commands and raw LiDAR scans directly to continuous steering and acceleration inputs.

> **Feedback:** "Interesting!"

---

## 2. Core Challenges
This project combines Natural Language Understanding (NLU) with agile vehicle control:

* **Multimodal Fusion:** The agent must fuse semantic information (text embeddings) with geometric perception (2D LiDAR point clouds).
* **Dynamic Objective Switching:** The optimal policy changes based on the command[cite: 15]. "Push" requires aggressive braking and clipping apexes, while "Conserve" requires coasting and smoother trajectories[cite: 15].
* **Vehicle Dynamics:** Unlike traffic navigation, the agent must operate at the limits of friction, where excessive steering or throttle inputs can lead to loss of traction[cite: 16].

---

## 3. Proposed Approach
I will implement a **Language-Conditioned Reinforcement Learning** architecture:

1. **Instruction Encoder:** A pre-trained Transformer (e.g., DistilBERT or MiniLM) will encode the race engineer's text commands into a latent semantic vector[cite: 19].
    > **Feedback:** "OK!"

2. **Perception Network:** A Multi-Layer Perceptron (MLP) or 1D-CNN will process the raw LiDAR range data to extract track boundaries and obstacle positions[cite: 21].
    > **Feedback:** "OK!"

3. **Policy Optimization:** A PPO (Proximal Policy Optimization) agent will take the concatenated state `[Lidar Features + Command Embedding]` and output continuous control actions (Steering $\delta$, Velocity $v$)[cite: 23].
    > **Feedback:** "This is going to be the most challenging, but interesting nevertheless..."

---

## 4. Tools & Technical Stack
> **Feedback:** "Indeed, you need a simulation-ready environment to focus on the agent!"

* **Simulation Environment:** The primary target environment is the **F1TENTH Gym**, a high-fidelity simulator for 1:10 scale autonomous racing vehicles featuring Ackermann steering and LiDAR perception[cite: 27].
    * *Note:* Due to potential compatibility constraints with Apple Silicon (ARM64) hardware, an alternative physics-based environment (e.g., Gymnasium CarRacing or Highway-Env) may be substituted if necessary to ensure stable local training[cite: 28].
    > **Feedback:** "Check out also CARLA!"

* **NLP/Embeddings:** Hugging Face Transformers. Specifically utilizing the `all-MiniLM-L6-v2` embedding model[cite: 29].

* **RL Framework:** Stable-Baselines3. Utilizing Custom Policy Networks to handle the multi-modal input space[cite: 30].
    > **Feedback:** "Nice combination of tools"

---

## Supervisor Concluding Remarks

* **Metrics:** You need to think also of some metrics that can quantify how well you satisfy the verbal guidelines, along with the driving constraints.
* **Rewards:** Of course, this will be reflected on the rewards you will provide to the RL agent.
* **Closing:** In any case, go ahead.. and good luck!