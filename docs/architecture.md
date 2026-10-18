# Boundary and extension points

`bridge.py` selects one Arena checkout and sets its resource root before importing it. `SkillBatch` shares Arena's `VectorArena`, which owns isolated persistent Rust processes. The simulator accepts bounded training initial conditions only during initialization. It exposes no mid-episode teleport action to the learned policy.

`tasks.py` owns seeded scenario generation, reward components, success conditions and time limits. All damage comes from ordinary projectile/armor collisions. `encoding.py` is the versioned policy observation/action contract. It reads only the controlled team's observation. The training host may inspect the other team's own telemetry to compute rewards, but it is excluded from policy features.

`envs.py` adapts the same batch to Gymnasium and SB3. `SingleRobotEnv` requires explicit reset after a terminal transition. `SkillVecEnv` saves the terminal observation and truncation flag before automatically resetting an actor for SB3. Actor subsets reset independently.

`learning.py` delegates PPO, GAE and optimization to Stable-Baselines3. It records checksums before/after optimization, the number of updates, source fingerprints, and actual sample counts. Evaluation uses new seeded episodes and deterministic actions. Parallel actor processes run on CPU; the first small MLP recipe uses CPU inference/training.

`hierarchy.py` freezes separate outpost/defend skill policies. A categorical manager observes legal local information and selects a skill. Each option drives up to ten real low-level decisions without reinitializing the simulator. Completed actors stop advancing while peers finish their current option. Scenario-controlled invaders continue regardless of the selected skill. An event reward is counted once. Terminal replay hooks run before an actor resets.

First-version limits: W3, red team, one learned robot, a single season, assisted ballistics, flat navigation goals, one ramp, local assault/interception curricula. Initial ammunition is supplied as an unranked training initial condition. No complete-map strategic performance claim, raw-image learning, torque control, or learned multi-robot coordination is made.

To extend: define a task with explicit start/goal and measurable physical success; add observation fields under a new schema version when needed; train and evaluate the skill against both a scripted feasibility controller and an untrained baseline; then make it selectable by the manager. Keep arena physics and rules changes in Arena-1.
