


# Gradient-based Optimization
## Gradient descent
Given Loss function $L(\boldsymbol{\theta})$, the gradient descent algorithm is shown as follows,

\begin{algorithm}[H]
\caption{Gradient descent}
\label{algo_GD}

Initialize weights $\boldsymbol{\theta}$, e.g. $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$

\While{\text{stopping criterion is NOT satisfied}}
{
    $\Delta \boldsymbol{\theta} = - \nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta})$ \\
    
    choose step size $\eta$ (this step is called line search)
    
    $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \Delta \boldsymbol{\theta}$ 
}
\end{algorithm}

One of the common stopping criterion is $||\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta})||_2 \leq \delta$. Gradient descent is a first-order gradient-based optimization algorithm.

## Stochastic and batch gradient descent
In gradient descent, the loss is the summation of all the data points with size $N$. This is time consuming if the data size is large. Thus we have stochastic gradient descent.

\begin{algorithm}[H]
\caption{Stochastic gradient descent}
\label{algo_GD}

Initialize weights $\boldsymbol{\theta}$, e.g. $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$

\While{\text{stopping criterion is NOT satisfied}}
{
    randomly pick one sample $(\mathbf{x_i}, y_i), \forall i \in [1,N]$ \\
    
    $\Delta \boldsymbol{\theta} = - \nabla_{\boldsymbol{\theta}}L_i(\boldsymbol{\theta})$ \\
    
    choose step size $\eta$ \\
    
    $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \Delta \boldsymbol{\theta}$ 
}
\end{algorithm}

We can also take a mini-batch of sample each iteration instead of just one sample. 

\begin{algorithm}[H]
\caption{Batch gradient descent}
\label{algo_GD}

Initialize weights $\boldsymbol{\theta}$, e.g. $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$

\While{\text{stopping criterion is NOT satisfied}}
{
    pick one batch of sample with size $B$. $([\mathbf{x_1}, y_1), \cdots (\mathbf{x_B}, y_B))]$ \\
    
    $L_B = \frac{1}{B} \sum_i L( \mathbf{x_i}, y_i, \boldsymbol{\theta})$ \\
    
    $\Delta \boldsymbol{\theta} = - \nabla_{\boldsymbol{\theta}}L_B(\boldsymbol{\theta})$ \\
    
    choose step size $\eta$ \\
    
    $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \Delta \boldsymbol{\theta}$ 
}
\end{algorithm}

## Modern gradient-based optimizers
### Momentum
For momentum, we update the weights of the weighted average of the past gradients. The update would be like
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_t$$
$$\mathbf{m}_t = \beta \mathbf{m}_{t-1} + (1-\beta) \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$$

### Adam
if the gradient is really large, then $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$ is large, then next time $\mathbf{g}_{t+1}$ is gonna be small because of $1-\beta$. 

Adam combines momentum and RMSProp.
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_{\boldsymbol{\theta}_t} L(\boldsymbol{\theta})$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta)_2 \nabla_{\boldsymbol{\theta}_t}^2 L(\boldsymbol{\theta})$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2}$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \frac{\eta}{\hat{\mathbf{v}}_t+\epsilon} \hat{\mathbf{m}}_t $$
