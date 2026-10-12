# Diffusion Models

(Blue or black color does not mean anything in this note)

## I. DDPM

<!-- PDF page 1 -->

Denoising process, called Stochastic Langevin Dynamics

$$
\mathbf{x}^{k+1}=\alpha\left[\mathbf{x}^k-\gamma\boldsymbol{\epsilon}_\theta(\mathbf{x}^k,k)+\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})\right]
$$

Gaussian noise added, key to make diffusion work

$$
\Updownarrow
$$

$$
\mathbf{x}'=\mathbf{x}-\gamma\nabla_{\mathbf{x}}E(\mathbf{x})
$$

noise prediction network $\boldsymbol{\epsilon}_\theta(\mathbf{x},k)$ effectively predicts gradient field $\nabla E(\mathbf{x})$

### 1.1 proof of this relationship

Based on Boltzmann distribution, the probability of an image

$$
p(\mathbf{x}_t)=\frac{\exp(-E(\mathbf{x}_t))}{\int\exp(-E(\mathbf{x}_t))\,dt}
=\frac{\exp(-E(\mathbf{x}_t))}{Z}
$$

take log (ln actually) on both sides:

$$
\log p(\mathbf{x}_t)=-E(\mathbf{x}_t)-0=-E(\mathbf{x}_t)
$$

<!-- PDF page 2 -->

then take gradient over $\mathbf{x}_t$, on both sides

$$
\boxed{\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)=-\nabla_{\mathbf{x}_t}E(\mathbf{x}_t)}
$$

which is called [score function]. Note this gradient is not w.r.t. model parameters $\theta$, but w.r.t. data images

Then, we talk about [forward] diffusion process.

we create a noisy image $\mathbf{x}_t$ by adding Gaussian noise $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$:

$$
\mathbf{x}_t=
\underbrace{\sqrt{\bar{\alpha}_t}}_{\text{noise schedule}}\mathbf{x}_0
+\underbrace{\sqrt{1-\bar{\alpha}_t}}_{\text{std of the noise}}\boldsymbol{\epsilon}
$$

rearrange to get noise $\boldsymbol{\epsilon}$

$$
\boldsymbol{\epsilon}
=\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1-\bar{\alpha}_t}}
=\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sigma_t}
$$

in the probabilistic language:

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)=\boldsymbol{\epsilon}
=\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sigma_t}
$$

hence

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)\sim\mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0,\sigma_t^2\mathbf{I})
$$

The gradient of the log probability of a Gaussian distrib

$$
\begin{aligned}
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
&=-\frac{\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sigma_t^2}\\
&=-\frac{\boldsymbol{\epsilon}}{\sigma_t}
\end{aligned}
$$

<!-- PDF page 3 -->

And the noise in DDPM is from a network parameterized with $\theta$

hence the noise is predicted from $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$.

And we need to do some calculation to connect to the previous equation.

Since using MSE trained, the network output is

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)=\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]
$$

take expectation on both sides.

$$
\begin{aligned}
\mathbb{E}\left[\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)\right]
&=-\frac{\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]}{\sigma_t}\\
&=-\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sigma_t}
\end{aligned}
$$

Then we try to connect $\log q(\mathbf{x}_t\mid\mathbf{x}_0)$ with $\log p(\mathbf{x}_t)$, as shown in the score function,

$p(\mathbf{x}_t)$ is the marginal density/prob

$$
p(\mathbf{x}_t)=\int p(\mathbf{x}_0)q(\mathbf{x}_t\mid\mathbf{x}_0)\,d\mathbf{x}_0
$$

take gradient over $\mathbf{x}_t$ on both sides

$$
\nabla_{\mathbf{x}_t}p(\mathbf{x}_t)
=\int p(\mathbf{x}_0)\nabla_{\mathbf{x}_t}q(\mathbf{x}_t\mid\mathbf{x}_0)\,d\mathbf{x}_0
$$

<!-- PDF page 4 -->

applying the log rule:

$$
\nabla_{\mathbf{x}}\log q(\mathbf{x})
=\frac{\nabla_{\mathbf{x}}q(\mathbf{x})}{q(\mathbf{x})}
$$

so we have

$$
\nabla_{\mathbf{x}_t}q(\mathbf{x}_t\mid\mathbf{x}_0)
=q(\mathbf{x}_t\mid\mathbf{x}_0)\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
$$

substitute back, we have,

$$
\nabla_{\mathbf{x}_t}p(\mathbf{x}_t)
=\int p(\mathbf{x}_0)\cdot q(\mathbf{x}_t\mid\mathbf{x}_0)
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
$$

$$
\begin{aligned}
\frac{\nabla_{\mathbf{x}_t}p(\mathbf{x}_t)}{p(\mathbf{x}_t)}
&=\int\frac{p(\mathbf{x}_0)q(\mathbf{x}_t\mid\mathbf{x}_0)}{p(\mathbf{x}_t)}
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)\\
&=\int p(\mathbf{x}_0\mid\mathbf{x}_t)\nabla_{\mathbf{x}_t}\log[q(\mathbf{x}_t\mid\mathbf{x}_0)]
\end{aligned}
$$

Bayes equa.

log trick

$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)
=\int p(\mathbf{x}_0\mid\mathbf{x}_t)\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
$$

Then According to the definition of Expectation,

$$
\mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0\mid\mathbf{x}_t)}[f(\mathbf{x}_0)]
=\int p(\mathbf{x}_0\mid\mathbf{x}_t)f(\mathbf{x}_0)\,d\mathbf{x}_0
$$

hence the right hand side is the $\mathbb{E}$ of $\nabla_{\mathbf{x}_t}q(\mathbf{x}_t\mid\mathbf{x}_0)$

Then,

$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)
=\mathbb{E}\left[\nabla_{\mathbf{x}_t}q(\mathbf{x}_t\mid\mathbf{x}_0)\right]
$$

score function; MSE network output

$$
-\nabla_{\mathbf{x}_t}E(\mathbf{x}_t)
=-\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sigma_t}
$$

<!-- PDF page 5 -->

so we have

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)=\sigma_t\nabla_{\mathbf{x}_t}E(\mathbf{x}_t)
$$

so the noise prediction network's output is (negatively) proportional to the score function.

### DDPM training loss function:

$$
\mathcal{L}=\mathrm{MSE}\left(\boldsymbol{\epsilon}^k,\boldsymbol{\epsilon}_\theta(\mathbf{x}^0+\boldsymbol{\epsilon}^k,k)\right)
$$

This is used to supervise the noise prediction, but there's something deeper in this loss, let's dive in.

Our objective to train a generative model is to generate images closer to the distribution of train data $\mathbf{x}_0$, thus

$$
\max_\theta\mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0)]
$$

(actually it's ln)

which is equivalent to

$$
\min_\theta\mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_0)}[-\log p_\theta(\mathbf{x}_0)]
$$

since the distribution of the data $\mathbf{x}_0$, $p_\theta(\mathbf{x}_0)$, is not tractable, we will minimize a bound of this log likelihood.

<!-- PDF page 6 -->

the probability is marginalized as, hence integrate every single intermediate noisy state

$$
p_\theta(\mathbf{x}_0)=\int p_\theta(\mathbf{x}_{0:T})\,d\mathbf{x}_{1:T}
$$

$$
\begin{aligned}
-\log p_\theta(\mathbf{x}_0)
&=-\log\int q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)
\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\,d\mathbf{x}_{1:T}\\
&=-\log\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right]
\end{aligned}
$$

Then apply Jensen's inequality:

$$
\leq\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[-\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right]
$$

next we visit back the Markov chain of forward process and denoising process, with $p_\theta$

reverse process:

$$
p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
$$

forward process:

$$
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^T q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
$$

substitute back to the previous equation, we have

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[-\log\frac{p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}
{\prod_{t=1}^T q(\mathbf{x}_t\mid\mathbf{x}_{t-1})}\right]\\
&=\mathbb{E}_{q(\mathbf{x}_{t:T}\mid\mathbf{x}_0)}
\left[-\log p(\mathbf{x}_T)-\sum_{t=1}^T
\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})}\right]
\end{aligned}
$$

<!-- PDF page 7 -->

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[-\log p(\mathbf{x}_T)-\sum_{t=2}^T
\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})}
-\log\frac{p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}{q(\mathbf{x}_1\mid\mathbf{x}_0)}\right]\\
&=\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[-\log p(\mathbf{x}_T)-\sum_{t=2}^T
\log\left[
\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
\cdot\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\right]
-\log\frac{p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}{q(\mathbf{x}_1\mid\mathbf{x}_0)}\right]
\end{aligned}
$$

(via joint prob)

let's unpack this part

$$
\begin{aligned}
\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)}
&=\log q(\mathbf{x}_1\mid\mathbf{x}_0)-\log q(\mathbf{x}_2\mid\mathbf{x}_0)\\
&\quad+\log q(\mathbf{x}_2\mid\mathbf{x}_0)-\log(q(\mathbf{x}_3\mid\mathbf{x}_0))\\
&\quad+\cdots\\
&=\log q(\mathbf{x}_1\mid\mathbf{x}_0)-\log q(\mathbf{x}_T\mid\mathbf{x}_0)
\end{aligned}
$$

substitute back:

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\Bigg[
-\log p(\mathbf{x}_T)-\log q(\mathbf{x}_T\mid\mathbf{x}_0)
-\sum_{t=2}^T\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}
{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}\\
&\hspace{5em}+\log q(\mathbf{x}_1\mid\mathbf{x}_0)
-\log\frac{p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}{q(\mathbf{x}_1\mid\mathbf{x}_0)}
\Bigg]\\
&=\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\left[
-\log\frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T\mid\mathbf{x}_0)}
-\sum_{t=2}^T\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}
{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\right]\\
&=\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\left[
\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}
+\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}
-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\right]
\end{aligned}
$$

<!-- PDF page 8 -->

Let's derive each term of the above equation:

(1).

$$
\begin{aligned}
\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}\right]
&=\mathbb{E}_{q(\mathbf{x}_T\mid\mathbf{x}_0)}
\left[\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}\right]\\
&=D_{\mathrm{KL}}\left[q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right]
\end{aligned}
$$

(let's unpack this!)

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}\right]\\
&=\int_{\mathbf{x}_1}\cdots\int_{\mathbf{x}_{T-1}}\int_{\mathbf{x}_T}
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)
\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}
\,d\mathbf{x}_T\,d\mathbf{x}_{T-1}\cdots d\mathbf{x}_1\\
&=\int_{\mathbf{x}_T}\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}
\left[\int_{\mathbf{x}_1}\cdots\int_{\mathbf{x}_{T-1}}
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\,d\mathbf{x}_{T-1}\cdots d\mathbf{x}_1\right]d\mathbf{x}_T
\end{aligned}
$$

(only related to $(\mathbf{x}_T,\mathbf{x}_0)$, so move to here)

$$
\int_{\mathbf{x}_1}\cdots\int_{\mathbf{x}_{T-1}}
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\,d\mathbf{x}_{T-1}\cdots d\mathbf{x}_1
=q(\mathbf{x}_T\mid\mathbf{x}_0)
$$

marginal distribution

$$
\mathbb{E}_{q(\mathbf{x}_T\mid\mathbf{x}_0)}
\left[\log\frac{q(\mathbf{x}_T\mid\mathbf{x}_0)}{p(\mathbf{x}_T)}\right]
=D_{\mathrm{KL}}\left[q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right]
$$

2) the 2nd term:

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\right]\\
&=\mathbb{E}_{q(\mathbf{x}_{t,t-1}\mid\mathbf{x}_0)}
\left[\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\right]
\end{aligned}
$$

only $(\mathbf{x}_t,\mathbf{x}_{t-1})$ matters

Since

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
=\frac{q(\mathbf{x}_t,\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)}
$$

<!-- PDF page 9 -->

then

$$
\begin{aligned}
&\mathbb{E}_{q(\mathbf{x}_t,\mathbf{x}_{t-1}\mid\mathbf{x}_0)}
\left[\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\right]\\
&=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\left[\mathbb{E}_{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
\sum_{t=2}^T\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}
{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\right]\\
&=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\left[\sum_{t=2}^T D_{\mathrm{KL}}
\left(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)\right]
\end{aligned}
$$

(3)

$$
\begin{aligned}
\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
\left[-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\right]
&=\mathbb{E}_{q(\mathbf{x}_1\mid\mathbf{x}_0)}
\left[-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\right]
\end{aligned}
$$

we define these loss terms in DDPM paper:

1.

$$
L_T=D_{\mathrm{KL}}\left(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right)
$$

2.

$$
L_{t-1}=D_{\mathrm{KL}}\left(q(t-1\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)
$$

3.

$$
L_0=-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)
$$

$L_T$ does not have any learnable parameter so we drop it from the loss

<!-- PDF page 10 -->

The KL divergence of two $d$-dim multivariate Gaussians $\mathcal{N}_0(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$ and $\mathcal{N}_1(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1)$ is

$$
\begin{aligned}
D_{\mathrm{KL}}(\mathcal{N}_0\|\mathcal{N}_1)
=\frac{1}{2}\Bigg[
&\operatorname{tr}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_0)
+(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^T
\boldsymbol{\Sigma}_1^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)\\
&-d+\log\left(\frac{\det\boldsymbol{\Sigma}_1}{\det\boldsymbol{\Sigma}_0}\right)
\Bigg]
\end{aligned}
$$

the distributions of $q$ and $p_\theta$ are also multivariate Gaussian

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\boldsymbol{\mu}}(\mathbf{x}_t,\mathbf{x}_0),\tilde{\beta}_t\mathbf{I})
$$

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\sigma_t^2\mathbf{I})
$$

$\tilde{\boldsymbol{\mu}}(\mathbf{x}_t,\mathbf{x}_0)$ are like a ground truth label

substitute back to the KL divergence $D_{\mathrm{KL}}(\mathcal{N}_0\|\mathcal{N}_1)$

$$
\operatorname{tr}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_0)
=\operatorname{tr}\left(\frac{\tilde{\beta}_t}{\sigma_t^2}\mathbf{I}\right)
=d\cdot\frac{\tilde{\beta}_t}{\sigma_t^2}
$$

$$
\log\left[\frac{(\sigma_t^2)^d}{(\tilde{\beta}_t)^d}\right]
=d\ln\left(\frac{\sigma_t^2}{\tilde{\beta}_t}\right)
$$

$$
(\boldsymbol{\mu}_\theta-\tilde{\boldsymbol{\mu}}_t)^T
(\sigma_t^2\mathbf{I})^+
(\boldsymbol{\mu}_\theta-\tilde{\boldsymbol{\mu}}_t)
=\frac{1}{\sigma_t^2}\|\boldsymbol{\mu}_\theta-\tilde{\boldsymbol{\mu}}_t\|^2
$$

So, the KL divergence of $p_\theta$ and $q$ is

$$
\begin{aligned}
D_{\mathrm{KL}}(q\|p_\theta)=L_{t-1}
&=\frac{1}{2}\left[
d\frac{\tilde{\beta}_t}{\sigma_t^2}
+\frac{1}{\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t-\boldsymbol{\mu}_\theta\|^2
-d+d\log\left(\frac{\sigma_t^2}{\tilde{\beta}_t}\right)\right]\\
&=\frac{1}{2\sigma_t^2}
\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)-\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)\|^2
+\mathrm{const}
\end{aligned}
$$

<!-- PDF page 11 -->

Here let's pause and visit back to forward and reverse diffusion processes.

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)
:=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\boldsymbol{\Sigma}_\theta(\mathbf{x}_t,t))
$$

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
:=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})
$$

Noise at $t$

$$
\alpha_t=1-\beta_t
$$

$$
=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\mathbf{x}_{t-1},(1-\alpha_t)\mathbf{I})
$$

thus

$$
\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1},
\qquad\boldsymbol{\epsilon}_{t-1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

one step back we have

$$
\mathbf{x}_{t-1}=\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}
$$

substitute back,

$$
\begin{aligned}
\mathbf{x}_t
&=\sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}
+\sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}\right)
+\sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}\\
&=\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}
+\underbrace{\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}
+\sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t}_{\text{new noise term}}
\end{aligned}
$$

<!-- PDF page 12 -->

adding these two noise distribution we have new variance:

$$
\begin{aligned}
\left(\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\right)^2
+\left(\sqrt{1-\alpha_t}\right)^2
&=\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)\\
&=1-\alpha_t\alpha_{t-1}
\end{aligned}
$$

thus $\mathbf{x}_t$ can be written as

$$
\mathbf{x}_t=\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}
+\sqrt{1-\alpha_t\alpha_{t-1}}\bar{\boldsymbol{\epsilon}}_{\mathrm{new}}
$$

(extend back to $\mathbf{x}_0$)

$$
\begin{aligned}
&=\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_T}\mathbf{x}_0
+\sqrt{1-\alpha_t\alpha_{t-1}\cdots\alpha_T}\bar{\boldsymbol{\epsilon}}_{\mathrm{new}}\\
&=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\bar{\boldsymbol{\epsilon}}_{\mathrm{new}}
\end{aligned}
$$

thus

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)
=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})
$$

where

$$
\bar{\alpha}_t=\prod_{k=1}^T\alpha_k
$$

Next we try to get $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$

<!-- PDF page 13 -->

According to bayes rules:

$$
\begin{aligned}
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
&=\frac{q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}
{q(\mathbf{x}_t\mid\mathbf{x}_0)}\\
&=\frac{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}
{q(\mathbf{x}_t\mid\mathbf{x}_0)}\\
&\propto q(\mathbf{x}_t\mid\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)
\end{aligned}
$$

(Markovian property)

where:

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})
$$

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)
=\mathcal{N}(\mathbf{x}_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0,(1-\bar{\alpha}_{t-1})\mathbf{I})
$$

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)
=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})
$$

The denominator does not have $\mathbf{x}_{t-1}$ and can be treated as a constant.

we have a new distribution as

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\boldsymbol{\mu}}(\mathbf{x}_t,\mathbf{x}_0),\tilde{\beta}_t\mathbf{I})
$$

it's a product of two Gaussian distributions

but $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ is about $\mathbf{x}_t$, so we need to manipulate it a bit.

<!-- PDF page 14 -->

$$
\begin{aligned}
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
&\propto\exp\left(-\frac{1}{2\beta_t}\|\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2\right)\\
&\propto\exp\left(-\frac{\alpha_t}{2\beta_t}
\left\|\mathbf{x}_{t-1}-\frac{\mathbf{x}_t}{\sqrt{\alpha_t}}\right\|^2\right)
\end{aligned}
$$

hence

$$
\mathbf{x}_{t-1}\sim\mathcal{N}\left(\frac{\mathbf{x}_t}{\sqrt{\alpha_t}},\frac{\beta_t}{\alpha_t}\mathbf{I}\right)
$$

Thus combine these two Gaussians, we get the variance

$$
\frac{1}{\tilde{\beta}_t}=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}
\qquad
\left(\frac{1}{\sigma_{\mathrm{new}}^2}=\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}\right)
$$

$$
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

and the mean

$$
\tilde{\boldsymbol{\mu}}_t
=\tilde{\beta}_t\left[
\frac{\frac{\mathbf{x}_t}{\sqrt{\alpha_t}}}{\frac{\beta_t}{\alpha_t}}
+\frac{\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0}{1-\bar{\alpha}_{t-1}}\right]
$$

$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)$

$$
\left(\boldsymbol{\mu}_{\mathrm{new}}
=\left(\frac{\boldsymbol{\mu}_1}{\sigma_1^2}+\frac{\boldsymbol{\mu}_2}{\sigma_2^2}\right)\cdot\sigma_{\mathrm{new}}^2\right)
$$

$$
=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0
+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

<!-- PDF page 15 -->

Recall our previous derivation, with abuse of notation

$$
\begin{aligned}
L_{t-1}
&=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\left[D_{\mathrm{KL}}\left(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)
\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)\right]\\
&=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\left[\frac{1}{2\sigma_t^2}
\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)-\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)\|^2\right]
+\mathrm{const}
\end{aligned}
$$

rewrite $\mathbf{x}_t$ as

$$
\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon})
=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},
\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

thus we have

$$
\begin{aligned}
L_{t-1}-\mathrm{const}
=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\Bigg[\frac{1}{2\sigma_t^2}\Bigg\|
&\tilde{\boldsymbol{\mu}}_t\left(
\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}),
\underbrace{\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon})
-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}\right)}_{\mathbf{x}_0}
\right)\\
&-\boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}),t)
\Bigg\|^2\Bigg]
\end{aligned}
$$

substitute this $\mathbf{x}_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$ into

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)
&=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}
\cdot\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}\right)\\
&\quad+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
\end{aligned}
$$

and

$$
\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}}=\frac{1}{\sqrt{\alpha_t}},
\qquad\text{since }\bar{\alpha}_t=\alpha\cdot\bar{\alpha}_{t-1}
$$

<!-- PDF page 16 -->

Combine the terms of $\mathbf{x}_t$

$$
\begin{aligned}
&\left(\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}
+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\right)\mathbf{x}_t\\
&=\frac{\beta_t+\alpha_t(1-\bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}\mathbf{x}_t\\
&=\frac{1-\alpha_t+\alpha_t-\bar{\alpha}_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}\mathbf{x}_t
=\frac{\mathbf{x}_t}{\sqrt{\alpha_t}}
\end{aligned}
$$

then the terms of $\boldsymbol{\epsilon}$

$$
-\frac{\beta_t\cdot\sqrt{1-\bar{\alpha}_t}}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}
\cdot\boldsymbol{\epsilon}
=-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}\cdot\sqrt{\alpha_t}}\boldsymbol{\epsilon}
$$

So combine them we have

$$
\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)
=\tilde{\boldsymbol{\mu}}(\mathbf{x}_t,\mathbf{x}_0)
$$

Then the loss term is

$$
\begin{aligned}
L_{t-1}+\mathrm{const}
=\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}
\Bigg[\frac{1}{2\sigma_t^2}\Bigg\|
&\frac{1}{\sqrt{\alpha_t}}\left(
\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon})
-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)\\
&-\boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon}),t)
\Bigg\|^2\Bigg]
\end{aligned}
$$

sin $\mathbf{x}_t=\mathbf{x}_t(\mathbf{x}_0,\boldsymbol{\epsilon})\sim q(\mathbf{x}_t\mid\mathbf{x}_0)$

<!-- PDF page 17 -->

The above expectation is

$$
\mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_0),\,\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},1)}
[\cdots]
=\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}[\cdots]
$$

$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)$ is the posterior mean of $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$

$\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ is the posterior mean of $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$

so we have, and parameterize $\boldsymbol{\epsilon}$

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)
=\frac{1}{\sqrt{\alpha_t}}\left(
\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)
$$

Then the loss function becomes

$$
L_{t-1}-\mathrm{const}
=\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}
\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}
\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2\right]
$$

with

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)
=\boldsymbol{\epsilon}_\theta\left(
\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},t\right)
$$

The loss can be simplified as

$$
\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}
\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2
$$

MSE loss we had before

<!-- PDF page 19 -->

## Flow Matching models.

The original paper is [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). The following derivation starts with a linear path.

### Noise, data, and a velocity field

In this section, time runs from noise at $t=0$ to data at $t=1$. Denote a noise sample by $\mathbf{z}$ and a training sample by $\mathbf{y}$:

$$
\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\qquad
\mathbf{y}\sim q(\mathbf{y}),
\qquad
t\sim\mathcal{U}[0,1].
$$

Sample $\mathbf{z}$ and $\mathbf{y}$ independently. A neural network $\mathbf{v}_\theta(\mathbf{x},t)\in\mathbb{R}^d$ predicts the velocity at position $\mathbf{x}\in\mathbb{R}^d$ and time $t$.

Generation follows an ordinary differential equation:

$$
\frac{d\hat{\mathbf{x}}_t}{dt}
=\mathbf{v}_\theta(\hat{\mathbf{x}}_t,t),
\qquad
\hat{\mathbf{x}}_0=\mathbf{z}.
$$

The goal is for the distribution of $\hat{\mathbf{x}}_1$ to match the data distribution. This ODE model is a continuous normalizing flow. [Paper, Section 2](https://arxiv.org/html/2210.02747v2#S2)

### A linear path and its training target

For each sampled pair $(\mathbf{z},\mathbf{y})$, construct

$$
\mathbf{x}_t=(1-t)\mathbf{z}+t\mathbf{y}.
$$

Its endpoints are

$$
\mathbf{x}_{t=0}=\mathbf{z},
\qquad
\mathbf{x}_{t=1}=\mathbf{y}.
$$

Keep the endpoints fixed and differentiate with respect to time:

$$
\frac{d\mathbf{x}_t}{dt}
=-\mathbf{z}+\mathbf{y}
=\mathbf{y}-\mathbf{z}.
$$

Thus, each pair supplies both an intermediate sample and a known target velocity. Write that target as

$$
\mathbf{U}_t=\mathbf{y}-\mathbf{z}.
$$

Conditioned on $\mathbf{y}$, the intermediate distribution is Gaussian for $0\leq t<1$:

$$
p_t(\mathbf{x}\mid\mathbf{y})
=\mathcal{N}\left(\mathbf{x};t\mathbf{y},(1-t)^2\mathbf{I}\right).
$$

Rearrange the interpolation:

$$
\mathbf{z}=\frac{\mathbf{x}_t-t\mathbf{y}}{1-t}.
$$

Substitute this into the velocity:

$$
\begin{aligned}
\mathbf{y}-\mathbf{z}
&=\mathbf{y}-\frac{\mathbf{x}_t-t\mathbf{y}}{1-t}\\
&=\frac{\mathbf{y}-\mathbf{x}_t}{1-t}.
\end{aligned}
$$

The corresponding conditional velocity field is therefore

$$
\mathbf{u}_t(\mathbf{x}\mid\mathbf{y})
=\frac{\mathbf{y}-\mathbf{x}}{1-t},
\qquad 0\leq t<1.
$$

Along the sampled path, $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{y})=\mathbf{y}-\mathbf{z}$. At $t=1$, the conditional distribution becomes a point mass at $\mathbf{y}$; the sampled-pair target remains $\mathbf{y}-\mathbf{z}$.

### Conditional Flow Matching training loss

Regress the predicted velocity onto the sampled-pair target:

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=\mathbb{E}_{t,\mathbf{z},\mathbf{y}}
\left[
\left\|
\mathbf{v}_\theta\left((1-t)\mathbf{z}+t\mathbf{y},t\right)
-(\mathbf{y}-\mathbf{z})
\right\|^2
\right]
$$

A training step consists of sampling a data batch, independent Gaussian noise, and one time per sample, then constructing $\mathbf{x}_t$ and minimizing this MSE. Training samples come directly from the interpolation, so an ODE solve is not needed during training.