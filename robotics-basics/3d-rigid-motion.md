## 3D Rigid-body Kinematics
Given a mobile robot in 3D space, which can be an abstraction of a drone, a UFO. It's configurations is a 3D position vector and 3D rotation matrix of the body frame $\mathcal{F}_B$ expressed in the world frame, or inertial frame $\mathcal{F}_W$.
$$
\mathbf{T}_{WB}
=
\begin{bmatrix}
\mathbf{R}_{WB} & \mathbf{t}_{WB} \\
\mathbf{0}^{\top} & 1
\end{bmatrix}
\in \text{SE}(3)
$$

Sometimes for simplicity we drop the frame notations but it's important to be clear about which frame or physical quantity is expressed in which frame.

To drive the robot, we apply linear velocity $\mathbf{v}$ and angular velocity $\boldsymbol{\omega}$ through its actuators.
$$
{}_{B}\boldsymbol{\xi}_{WB}
=
\begin{bmatrix}
{}_{B}\mathbf{v}_{WB} \\
{}_{B}\boldsymbol{\omega}_{WB}
\end{bmatrix}
\in \mathbb{R}^{6}
$$

Note here we need three frame notations to denote linear and angular velocities (see [this great blog post](https://rpg.ifi.uzh.ch/docs/teaching/2025/FurgaleTutorial.pdf)), which means velocity of the robot body relative to its world/inertial frame, expressed in its body frame.

One example is that we can re-express the angular velocity of a body in relative to the world, and expressed in body frame, in the world frame through rotation $\mathbf{R}_{WB}$. And a simplied notation could be $\boldsymbol{\omega}_B$ and $B$ means this physical quantity is expressed in the body frame. By default it says this is the velocity of the body in relative to the world/inertial frame. Some literatures use ${}_{B}\boldsymbol{\omega}$ but the right subscript style is typically easier to process mentally (and code!) when doing calculation.

$$
\begin{gathered}
{}_{\underline{W}}\boldsymbol{\omega}_{WB}
\leftarrow \mathbf{R}_{\underline{WB \, {}_{B}}}\boldsymbol{\omega}_{WB} \\[0.75em]
\Downarrow \quad \text{simplified} \\[0.75em]
\boldsymbol{\omega}_{W} \leftarrow \mathbf{R}_{WB} \, \boldsymbol{\omega}_{B}
\end{gathered}
$$

The robot state at time $t$ is often represented by it's pose, linear and angular velocities,
$$
\mathbf{x}_t = \left\{\mathbf{T}_{WB}, \mathbf{v}_{B}, \boldsymbol{\omega}_{B} \right\}
$$

To associate the velocities with the robot pose $\mathbf{T} \in \text{SE}(3)$, we convert these into the Lie Algebras $\boldsymbol{\xi}^{\wedge} \in \mathfrak{se}(3)$,
$$
\boldsymbol{\xi}^{\wedge}
=
\begin{bmatrix}
\boldsymbol{\omega}^{\wedge} & \mathbf{v} \\
\mathbf{0}^{\top} & 0
\end{bmatrix}
\in \mathfrak{se}(3)
$$

$$
\boldsymbol{\omega}^{\wedge}
=
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
\in \mathfrak{so}(3)
$$

The new pose in the next time step is, use righ-hand update rule,
$$
\mathbf{T}_{t+1}
=
\mathbf{T}_{t}
\exp\left(
\boldsymbol{\xi}^{\wedge}\Delta t
\right)
$$

For simplicity, we can also express this as,
$$
\mathbf{p}_{t+1}
=
\mathbf{p}_{t}
+
\mathbf{R}_{t}\mathbf{v}\Delta t
$$

$$
\mathbf{R}_{t+1}
=
\mathbf{R}_{t}
\exp\left(
\boldsymbol{\omega}^{\wedge}\Delta t
\right)
$$

Next we introduce a concept called Adjoint, which is the linear mapping that changes a twist $\boldsymbol{xi}$ from one coordinate frame to another using the full rigid-body transform.

Give 3D transform $\mathbf{T}_{WB} \in \text{SE}(3)$, the Adjoint of $\mathbf{T}_{WB}$ transfer 3D twist from the body frame to the world frame,
$$
\boldsymbol\xi_W
=
\operatorname{Ad}_{\mathbf{T}_{WB}}\boldsymbol\xi_B
$$

Adjoint of $\mathbf{T}_{WB}$ is,
$$
\operatorname{Ad}_{\mathbf{T}_{WB}}
=
\begin{bmatrix}
\mathbf{R}_{WB} & [\mathbf{t}_{WB}]_{\times}\mathbf{R}_{WB} \\
\mathbf{0} & \mathbf{R}_{WB}
\end{bmatrix}
$$

and its inverse,
$$
\operatorname{Ad}_{T_{WB}}^{-1}
=
\operatorname{Ad}_{T_{BW}}
=
\begin{bmatrix}
\mathbf{R}_{WB}^{\top} & -\mathbf{R}_{WB}^{\top}[\mathbf{t}_{WB}]_{\times} \\
\mathbf{0} & \mathbf{R}_{WB}^{\top}
\end{bmatrix}
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
