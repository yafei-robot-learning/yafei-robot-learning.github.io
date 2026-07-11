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
