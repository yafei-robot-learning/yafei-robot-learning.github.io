


# Gradient-based Optimization
## Gradient descent
Given Loss function $L(\boldsymbol{\theta})$, the gradient descent algorithm is shown as follows, where $\eta$ is the learning rate, `loss_criterion` could be `nn.MSELoss()`.

```python
def gradient_descent(loss_criterion, theta, eta, delta):
    while True:
        loss = loss_criterion(theta)
        loss.backward()

        with torch.no_grad():
            grad = theta.grad
            if grad.norm() <= delta:
                break
            theta -= eta * grad

        theta.grad.zero_()

    return theta
```

One of the common stopping criterion is $||\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta})||_2 \leq \delta$. Gradient descent is a first-order gradient-based optimization algorithm.

## Stochastic and batch gradient descent
In gradient descent, the loss is the summation of all the data points with size $N$. This is time consuming if the data size is large. Thus we have stochastic gradient descent.

```python
def stochastic_gradient_descent(loss_criterion, theta, X, y, eta, delta):
    N = X.shape[0]
    while True:
        i = torch.randint(0, N, (1,)).item()
        loss = loss_criterion(theta, X[i], y[i])
        loss.backward()

        with torch.no_grad():
            grad = theta.grad
            if grad.norm() <= delta:
                break
            theta -= eta * grad

        theta.grad.zero_()

    return theta
```

We can also take a mini-batch of sample each iteration instead of just one sample. 

```python
def batch_gradient_descent(loss_criterion, theta, X, y, B, eta, delta):
    N = X.shape[0]
    while True:
        idx = torch.randint(0, N, (B,))
        loss = loss_criterion(theta, X[idx], y[idx])
        loss.backward()

        with torch.no_grad():
            grad = theta.grad
            if grad.norm() <= delta:
                break
            theta -= eta * grad

        theta.grad.zero_()

    return theta
```

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
