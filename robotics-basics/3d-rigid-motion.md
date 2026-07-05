## 3D Rigid-body Kinematics
Given a mobile robot in 3D space, which can be an abstraction of a drone, a UFO. It's configurations is a 3D position vector and 3D rotation matrix of the body frame $\mathcal{F}_B$ expressed in the world frame, or inertial frame $\mathcal{F}_W$.
$$
\mathbf{T}_{WB}
=
\begin{bmatrix}
\mathbf{R}_{WB} & \mathbf{p}_{WB} \\
\mathbf{0}^{\top} & 1
\end{bmatrix}
\in \text{SE}(3)
$$

Sometimes for simplicity we drop the frame notations but it's important to be clear about which frame or physical quantity is expressed in which frame.

To drive the robot, we apply linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$ through its actuators.
$$
\boldsymbol{\xi}_{B}
=
\begin{bmatrix}
\mathbf{v}_{B} \\
\boldsymbol{\omega}_{B}
\end{bmatrix}
\in \mathbb{R}^{6}
$$

Here $\mathbf{v}_{B}$ and $\boldsymbol{\omega}_{B}$ are the linear and angular velocities of the body relative to the world, expressed in the body frame. The relative frames are omitted because they are clear from context; the right subscript indicates the frame in which the vector is expressed (see [this great tutorial](https://rpg.ifi.uzh.ch/docs/teaching/2025/FurgaleTutorial.pdf) for the full three-frame notation).

For example, we can re-express the angular velocity from the body frame in the world frame using $\mathbf{R}_{WB}$:

$$
\boldsymbol{\omega}_{W}
=
\mathbf{R}_{WB}\boldsymbol{\omega}_{B}.
$$

The robot state at time $t$ is often represented by it's pose, linear and angular velocities,
$$
\mathbf{x}_t = \left\{\mathbf{T}_{WB}, \mathbf{v}_{B}, \boldsymbol{\omega}_{B} \right\}
$$

To associate the body-frame velocities with the robot pose $\mathbf{T} \in \text{SE}(3)$, we convert them into the Lie algebra $\boldsymbol{\xi}_{B}^{\wedge} \in \mathfrak{se}(3)$,
$$
\boldsymbol{\xi}_{B}^{\wedge}
=
\begin{bmatrix}
\boldsymbol{\omega}_{B}^{\wedge} & \mathbf{v}_{B} \\
\mathbf{0}^{\top} & 0
\end{bmatrix}
\in \mathfrak{se}(3)
$$

$$
\boldsymbol{\omega}_B ^{\wedge}
=
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
\in \mathfrak{so}(3)
$$

The new pose in the next time step is,
$$
\mathbf{T}_{t+1}
=
\mathbf{T}_{t}
\exp\left(
\boldsymbol{\xi}_{B}^{\wedge}\Delta t
\right)
$$

For simplicity, we can also express this as,
$$
\mathbf{p}_{t+1}
=
\mathbf{p}_{t}
+
\mathbf{R}_{t}\mathbf{v}_{B}\Delta t
$$

$$
\mathbf{R}_{t+1}
=
\mathbf{R}_{t}
\exp\left(
\boldsymbol{\omega}_{B}^{\wedge}\Delta t
\right)
$$

## 3D Rigid-body Dynamics

Kinematics describes how the pose changes given the linear and angular velocity. Next we talk about dynamics. Dynamics describes how forces and torques change that velocity. We assume that the origin of the body frame is at the center of mass, the mass $m$ is constant, and the inertia matrix $\mathbf{I}_B$ is expressed in the body frame.

### Translational dynamics

Newton's second law tells us,
$$
\mathbf{f}_{W}
=
m\dot{\mathbf{v}}_{W}.
$$

The body-frame velocities can be transformed to world-frame velocity by simply using rotation $\mathbf{R}_{WB}$, since $\mathbf{v}$ is a free vector, not a 3D point vector which is affected by the translational part of the pose $\mathbf{p}_{WB}$
$$
\mathbf{v}_{W}
=
\mathbf{R}_{WB}\mathbf{v}_{B}.
$$
Differentiating both sides and using
$$
\dot{\mathbf{R}}_{WB} = \mathbf{R}_{WB}\boldsymbol{\omega}_{B}^{\wedge} = \mathbf{R}_{WB}\left[\boldsymbol{\omega}_{B}\right]_{\times}
$$

gives,
$$
\begin{aligned}
\dot{\mathbf{v}}_{W}
&= \mathbf{R}_{WB}\left[\boldsymbol{\omega}_{B}\right]_{\times}\mathbf{v}_{B}
+ \mathbf{R}_{WB}\dot{\mathbf{v}}_{B} \\
&= \mathbf{R}_{WB}
\left(
\left[\boldsymbol{\omega}_{B}\right]_{\times}\mathbf{v}_{B}
+ \dot{\mathbf{v}}_{B}
\right)
\end{aligned}
$$

Substituting this result into Newton's second law gives
$$
\mathbf{f}_{W}
=
m\mathbf{R}_{WB}
\left(
\dot{\mathbf{v}}_{B}
+
\left[\boldsymbol{\omega}_{B}\right]_{\times}\mathbf{v}_{B}
\right).
$$
Left-multiplying by $\mathbf{R}_{BW}=\mathbf{R}_{WB}^{\top}$ and using $\mathbf{f}_{B}=\mathbf{R}_{BW}\mathbf{f}_{W}$ produces the body-frame translational equation
$$
\mathbf{f}_{B}
=
m
\left(
\dot{\mathbf{v}}_{B}
+
\left[\boldsymbol{\omega}_{B}\right]_{\times}\mathbf{v}_{B}
\right).
$$
Equivalently, the linear acceleration is
$$
\dot{\mathbf{v}}_{B}
=
\frac{1}{m}\mathbf{f}_{B}
-
\left[\boldsymbol{\omega}_{B}\right]_{\times}\mathbf{v}_{B}.
$$

Solving this ODE we are able to get the body-frame velocity given the force applied in the body frame. Numerically, we integrate this ODE with an initial velocity to solve the velocity $\mathbf{v}_B$


### Rotational dynamics
In rotational dynamics, we introduce torque $\boldsymbol{\tau}$, which is the rotational counterpart of force. It is the derivative of angular momentum $\mathbf{h}$ over time (like force is the derivative of linear momentum):
$$
\boldsymbol{\tau}_W
=
\frac{d\mathbf{h}_W}{dt}
$$

connect the augular momentum in world/inertial and body frame:
$$
\mathbf{h}_W
=
\mathbf{R}_{WB} \mathbf{h}_B
$$

take derivative at both sides,
$$
\dot{\mathbf{h}}_W = \mathbf{R}_{WB} \left(\dot{\mathbf{h}}_B + \left[\boldsymbol{\omega}_{B}\right]_{\times} \mathbf{h}_B \right)
$$

hence,
$$
\boldsymbol{\tau}_W = \mathbf{R}_{WB} \left(\dot{\mathbf{h}}_B + \left[\boldsymbol{\omega}_{B}\right]_{\times} \mathbf{h}_B \right)
$$

multiply rotation $\mathbf{R}_{WB}^\top$ on both sides,
$$
\mathbf{R}_{BW}\boldsymbol{\tau}_W
=
\dot{\mathbf{h}}_B
+
\left[\boldsymbol{\omega}_B\right]_{\times}\mathbf{h}_B
$$

The left-hand side is the world-frame torque re-expressed in the body frame:
$$
\boldsymbol{\tau}_B
=
\dot{\mathbf{h}}_B
+
\left[\boldsymbol{\omega}_B\right]_{\times}\mathbf{h}_B
$$

the body frame torque and it's derivative is
$$
\begin{aligned}
\mathbf{h}_B
&=
\mathbf{I}_B\boldsymbol{\omega}_B, \\
\dot{\mathbf{h}}_B
&=
\mathbf{I}_B\dot{\boldsymbol{\omega}}_B
\end{aligned}
$$

where $\mathbf{I}_B$ is the inertia matrix, which encodes how the mass is distributed in three dimensions around the body-frame origin.

which gives,
$$
\boldsymbol{\tau}_B
=
\mathbf{I}_B\dot{\boldsymbol{\omega}}_B
+
\left[\boldsymbol{\omega}_B\right]_{\times}
\mathbf{I}_B\boldsymbol{\omega}_B
$$

then
$$
\dot{\boldsymbol{\omega}}_B
=
\mathbf{I}_B^{-1}
\left(
\boldsymbol{\tau}_B
-
\left[\boldsymbol{\omega}_B\right]_{\times}
\mathbf{I}_B\boldsymbol{\omega}_B
\right)
$$

And now we have a similar ODE as we had in the translational dynamics.

### Newton-Euler dynamics
With the translational and rotational dynamics equations,
$$
\mathbf f_B
=
m\left(
\dot{\mathbf v}_B
+
[\boldsymbol\omega_B]_\times \mathbf v_B
\right)
$$

$$
\boldsymbol\tau_B
=
\mathbf I_B\dot{\boldsymbol\omega}_B
+
[\boldsymbol\omega_B]_\times
\mathbf I_B\boldsymbol\omega_B
$$

we stack linear and angular velocity into rigid-body velocity (twist), and force and torque into wrench, expressed in body frame:
$$
\boldsymbol\xi_B
=
\begin{bmatrix}
\mathbf v_B \\
\boldsymbol\omega_B
\end{bmatrix},
\qquad
\mathbf w_B
=
\begin{bmatrix}
\mathbf f_B \\
\boldsymbol\tau_B
\end{bmatrix}
$$

then combine the translational and rotational dynamics into one joint equation,
$$
\mathbf w_B
=
\begin{bmatrix}
m\mathbf I_3 & \mathbf 0 \\
\mathbf 0 & \mathbf I_B
\end{bmatrix}
\begin{bmatrix}
\dot{\mathbf v}_B \\
\dot{\boldsymbol\omega}_B
\end{bmatrix}
+
\begin{bmatrix}
m[\boldsymbol\omega_B]_\times \mathbf v_B \\
[\boldsymbol\omega_B]_\times \mathbf I_B\boldsymbol\omega_B
\end{bmatrix}
$$
here $\mathbf{I}_3$ is the $3 \times 3$ identity matrix.

or in matrix form,
$$
\mathbf M\dot{\boldsymbol\xi}_B
+
\mathbf C(\boldsymbol\xi_B)\boldsymbol\xi_B
=
\mathbf w_B
$$

with 6D rigid-body mass matrix:
$$
\mathbf M
=
\begin{bmatrix}
m\mathbf I_3 & \mathbf 0 \\
\mathbf 0 & \mathbf I_B
\end{bmatrix}
$$

velocity-dependent correction term:
$$
\mathbf C(\boldsymbol\xi_B)\boldsymbol\xi_B
=
\begin{bmatrix}
m[\boldsymbol\omega_B]_\times \mathbf v_B \\
[\boldsymbol\omega_B]_\times \mathbf I_B\boldsymbol\omega_B
\end{bmatrix}
$$

and the twist can be solved as,
$$
\dot{\boldsymbol\xi}_B
=
\mathbf M^{-1}
\left(
\mathbf w_B
-
\mathbf C(\boldsymbol\xi_B)\boldsymbol\xi_B
\right)
$$
