## Manipulator Kinematics
### Forward Kinematics

$$
\begin{aligned}
\mathbf{T}_{ab}(\theta)
&= \exp\left(\boldsymbol{\xi}_a^{\wedge}\theta \right)\mathbf{T}_{ab}(0) \\
&= \mathbf{T}_{ab}(0) \exp\left(\boldsymbol{\xi}_b^{\wedge}\theta \right)
\end{aligned}
$$
Depending on the frame in which the twist $\boldsymbol{\xi}$ is expressed, its exponential multiplies $\mathbf{T}_{ab}(0)$ on the left or the right: a spatial-frame twist multiplies on the left, whereas a body-frame twist multiplies on the right.

If $\boldsymbol{\xi}$ corresponds to a prismatic joint, then $\theta\in\mathbb{R}$ is the amount of translation; otherwise, $\theta\in S^1$ measures the angle of rotation about the axis.

The forward kinematics map is given as $\mathbf{T}_{ab}:\mathcal{Q}\rightarrow\mathrm{SE}(3)$, which is a mapping of joint configuration space $\mathcal{Q}$ to $\mathrm{SE}(3)$.

$$
\mathbf{T}_{WE}(\boldsymbol{\theta})
= \left( \prod_{i=1}^{N}\exp\left(\boldsymbol{\xi}_{Wi}^{\wedge}\theta_i\right) \right) \mathbf{T}_{WE}(\mathbf{0}).
$$

where $N$ is the number of joints, $\boldsymbol{\theta} = \left[ \theta_1, \theta_2, \cdots, \theta_N \right]^\top$ is the joint configurations. $W$ denotes the world/inertial frame and often the frame of the first link, and $E$ is the end-effector frame. $\mathbf{T}_{WE}(\mathbf{0})$ is called home configuration.

### Inverse Kinematics

Given the forward kinematics map $\mathbf{T}_{WE}:\mathcal{Q}\rightarrow\mathrm{SE}(3)$ and a desired end-effector pose $\mathbf{T}_d\in\mathrm{SE}(3)$, inverse kinematics (IK) is the problem to solve $\boldsymbol{\theta}\in\mathcal{Q}$ given

$$
\mathbf{T}_{WE}(\boldsymbol{\theta})=\mathbf{T}_d
$$

This equation may have a unique solution, multiple solutions, or no solution at all.

## Manipulator Dynamics

Before getting into the dynamics of a multi-joint manipulator system, we visit back to physics.



For a fixed-base manipulator with $N$ joints, the equations of motion including contact forces are

$$
\mathbf{M}(\boldsymbol{\theta})\ddot{\boldsymbol{\theta}}
+ \mathbf{C}(\boldsymbol{\theta},\dot{\boldsymbol{\theta}})\dot{\boldsymbol{\theta}}
+ \mathbf{g}(\boldsymbol{\theta})
= \boldsymbol{\tau}
+ \sum \mathbf{J}(\boldsymbol{\theta})^{\top}\mathbf{w}
$$

where:

$\boldsymbol{\theta},\dot{\boldsymbol{\theta}},\ddot{\boldsymbol{\theta}}\in\mathbb{R}^{N}$ are the joint positions (rad), angular velocities (rad/$s$), and angular accelerations (rad/$s^2$) (using local coordinates for joint angles).

$\mathbf{M}(\boldsymbol{\theta})\in\mathbb{R}^{N\times N}$ is the joint-space inertia matrix, determined by the mass and mass distributions of the links.

$\mathbf{C}(\boldsymbol{\theta},\dot{\boldsymbol{\theta}})\dot{\boldsymbol{\theta}}$ contains the Coriolis and centrifugal terms.

$\mathbf{g}(\boldsymbol{\theta})\in\mathbb{R}^{N}$ contains the gravitational terms.

$\boldsymbol{\tau}\in\mathbb{R}^{N}$ is the actuator torques (or forces for prismatic joints).

$\mathbf{w}\in\mathbb{R}^{m}$ stacks the external wrench exerted by the environment on the robot, e.g. contact force.

$\mathbf{J}(\boldsymbol{\theta})\in\mathbb{R}^{m\times N}$, the Jacobian, maps joint velocities (in joint space) to end-effector velocities (in task space); its transpose maps external wrench to joint torques.
