
# Clustering and Dimensionality Reduction
## PCA
Principal Component Analysis (PCA) is one of the most essential dimensionality reduction algorithm.

Given data matrix $\mathbf{X} \in \mathbb{R}^{N \times M}$, where $M$ represents the number of sample and $N$ number of dimension of each data point $\mathbf{x} \in \mathbb{R}^{N \times 1}$,

$$\mathbf{X} = 
\left[
\begin{array}{cccc}
\vertbar &\vertbar &       & \vertbar \\
\mathbf{x}_1, &\mathbf{x}_2, &\ldots, &\mathbf{x}_M \\
\vertbar &\vertbar &       & \vertbar \\
\end{array}
\right]$$

The PCA procedures are,

1) **Mean subtraction**

First compute the mean of data $X$, denoted as $\boldsymbol{\mu}$.
$$\boldsymbol{\mu} = \frac{1}{M} \sum_{i=1}^M \mathbf{x}_i$$

then subtract the mean for each data sample $\mathbf{x}_i, \hspace{0.1cm} \forall i \in [1,M]$
$$\bar{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu} = 
\left[
\begin{array}{cccc}
\vertbar &\vertbar &       & \vertbar \\
\mathbf{x}_1 - \boldsymbol{\mu}, &\mathbf{x}_2-\boldsymbol{\mu}, &\ldots, &\mathbf{x}_M-\boldsymbol{\mu} \\
\vertbar &\vertbar &       & \vertbar \\
\end{array}
\right]$$

\vspace{0.3cm}

2) **Find principal components via eigen decomposition**

Compute the covariance matrix and do eigen decomposition
$$\boldsymbol{\Sigma} =  \mathbb{E} \left[ (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x}- \boldsymbol{\mu})^{\top} \right] =  \bar{\mathbf{X}} \bar{\mathbf{X}}^{\top}  $$
$$\mathbf{V} \boldsymbol{\Lambda} \mathbf{V^{-1}} = \mathbf{\Sigma} $$

where the diagonal element of matrix $\boldsymbol{\Lambda}$ contains all the eigen values of $\mathbf{\Sigma}$, and $\mathbf{V}$ contains all the eigen vectors. Since $\mathbf{\Sigma}$ is a $N \times N$ matrix thus it has $N$ eigen vectors and $N$ eigen values. 

Next choose the eigen vectors $\mathbf{v}$ corresponding to the largest $P$ eigen values in $\boldsymbol{\Lambda}$ as the best direction for data to be projected on. We can form the projection matrix $\mathbf{B} \in \mathbb{R}^{N \times P}$ as,

$$\mathbf{B} = \left[
\begin{array}{cccc}
\vertbar &\vertbar &       & \vertbar \\
\mathbf{v}_1, &\mathbf{v}_2, &\ldots, &\mathbf{v}_P \\
\vertbar &\vertbar &       & \vertbar \\
\end{array}
\right]$$

\vspace{0.3cm}

3) **Projection and reconstruction**

Use the eigen vector $\mathbf{B}$ to project all data points. The projected points are computed as 
$$ \mathbf{X}' = \mathbf{B} ^{\top} \bar{\mathbf{X}}$$
We can reconstruct the projection as, 
$$ \mathbf{X}^* = \mathbf{B} \mathbf{B} ^{\top} \bar{\mathbf{X}} + \boldsymbol{\mu}$$


## Gaussian Mixture Model
Gaussian Mixture Model (GMM) is a density model which is a combination of $K$ (multivariate) Gaussian distributions. The Likelihood (the conditional probability of $\mathbf{x}$ given parameter $\boldsymbol{\theta}$ of $\mathbf{x}$ is obtained as,
$$p(\textbf{x}|\boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\textbf{x}| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\boldsymbol{\theta} = \{ \boldsymbol{\theta}_1, \cdots, \boldsymbol{\theta}_K\}$, and $\boldsymbol{\theta}_k = \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k \}, \forall k \in [1,K]$.

The weights $\pi_k$ is the probability that the distribution $\mathcal{N}(\textbf{x}| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ is selected. Hence
$$0 \leq \pi_k \leq 1 \text{, and } \sum_{k=1}^{K}\pi_k=1$$

Now given a set of data (fulfils i.i.d.), denoted as $\mathbf{X} =[\mathbf{x}_1, \cdots, \mathbf{x}_N]$. The Likelihood of $\mathbf{X}$ is,

$$p(\mathbf{X}|\boldsymbol{\theta}) = \prod_{n=1}^N p(\textbf{x}_n|\boldsymbol{\theta})$$

$$\text{with } p(\textbf{x}_n|\boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\textbf{x}_n| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \sum_{k=1}^{K} p(\textbf{x}_n|\boldsymbol{\theta}_k) \quad \text{(sum rule)}$$

Our goal is to find the parameter $\boldsymbol{\theta}$ given all the data observations $\mathbf{X}$.

First we obtain the log-likelihood as,
$$\ln p(\mathbf{X}|\boldsymbol{\theta}) = \sum_{n=1}^N \ln p(\textbf{x}_n|\boldsymbol{\theta}) $$

So in order to compute the optimal parameter $\boldsymbol{\theta}_{ML}$, we use a Maximum Likelihood approach, which is take the gradient of log-likelihood $L = \ln p(\mathbf{X}|\boldsymbol{\theta}) $ over $\boldsymbol{\theta}_k = \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k \}, \forall k \in [1,K]$

$$
\begin{aligned}
&\frac{\partial L}{\partial \boldsymbol{\mu}_k} = \sum_{n=1}^N \frac{\partial \ln p(\textbf{x}_n|\boldsymbol{\theta})}{\partial \boldsymbol{\mu}_k} = \mathbf{0}  \\
&\frac{\partial L}{\partial \boldsymbol{\Sigma}_k} = \sum_{n=1}^N \frac{\partial \ln p(\textbf{x}_n|\boldsymbol{\theta})}{\partial \boldsymbol{\Sigma}_k} = \mathbf{0}  \\
&\frac{\partial L}{\partial \pi_k} = \sum_{n=1}^N \frac{\partial \ln p(\textbf{x}_n|\boldsymbol{\theta})}{\partial \pi_k} =0
\end{aligned}
$$

However, we cannot obtain a close-form solution for these equations. In fact, we will be using an iterative approach called expectation maximization (EM) to find optimal parameters $\boldsymbol{\theta}_{ML}$. The idea of EM is to update one parameter in $\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k \}$ and keep all the others fixed.

Before formally giving the steps of EM algorithm. We define the responsibility of the $k$th mixture component for data $\textbf{x}_n$ as,
$$r_{nk} = \frac{\pi_k \mathcal{N}(\textbf{x}_n| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\textbf{x}_n| \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} = \frac{p(\textbf{x}_n|\boldsymbol{\theta}_k)}{p(\textbf{x}_n|\boldsymbol{\theta})}$$

The EM algorithm is listed as following,

1) Initialize parameters $\boldsymbol{\theta}_k, \forall k \in[1, K]$

2) E-step: Evaluate responsibility $r_{nk}$ for every data $\mathbf{x}_n, \forall n \in[1, N]$ with $\boldsymbol{\theta}_k, \forall k \in[1, K]$

3) Update $\boldsymbol{\theta}_k, \forall k \in[1, K]$ with the following update rule (proof of these rules pl refer to MMBook) with $r_{nk}$ computed in the previous step.

Update of mean $\boldsymbol{\mu}_k$ $$\boldsymbol{\mu}_k = \frac{\sum_{n=1}^{N}r_{nk}\mathbf{x}_n}{\sum_{n=1}^{N}r_{nk}}$$

Update of covariance matrix $\boldsymbol{\Sigma}_k$
$$\boldsymbol{\Sigma}_k = \frac{\sum_{n=1}^Nr_{nk}(\textbf{x}_n - \boldsymbol{\mu}_k)(\textbf{x}_n - \boldsymbol{\mu}_k)^{\top}}{\sum_{n=1}^N r_{nk}} $$

Update of weight $\pi_k$
$$\pi_k = \frac{\sum_{n=1}^N r_{nk}}{N}$$
