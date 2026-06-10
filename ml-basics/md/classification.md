# Models for Classification

## Logistic regression

$$p(y=1|\mathbf{x}, \boldsymbol{\theta}) = \sigma(\boldsymbol{\theta}^{\top} \mathbf{x})$$

where $\sigma(.)$ is the sigmoid function. $$\sigma(x) = \frac{1}{1+e^{-x}}$$

![Sigmoid function logistic curve](../figs/sigmoid.png "width=50%")

Sigmoid function properties:
\begin{itemize}
    \item Bounded between 0 and 1, thus can be interpreted as probability
    \item Monotonically increasing ← thus, usable to derive classification rules, e.g.
    \begin{itemize}
        \item $\sigma(x) \geq 0.5$, positive (classify as ’1’)
        \item $\sigma(x) < 0.5$, negative (classify as ’0’)
    \end{itemize}
\end{itemize}

Next we derive the loss function.
For a single sample $(\mathbf{x}_i, y_i)$, where $\mathbf{x}, \boldsymbol{\theta} \in \mathbb{R} ^{N \times 1}$

$$p(y_i|\mathbf{x}_i, \boldsymbol{\theta}) = \begin{cases}
      \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) & \text{if $y_i=1$ }\\
      1-\sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) & \text{otherwise}
    \end{cases}  
$$

since $y_i$ is either 0 or 1, we can rewrite this as
$$p(y_i|\mathbf{x}_i, \boldsymbol{\theta}) = \left[ \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right] ^{y_i} \left[1- \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right] ^{1-y_i}$$

extend to whole dataset we define energy function which is the total likelihood, and the size of the dataset is $M$, hence 
$\mathbf{X} \in \mathbb{R} ^{N \times M}$, and $\mathbf{y} \in \mathbb{R} ^{M \times 1}$ 
$$
\begin{aligned}
E = \prod_{i=1}^{M} p(y_i|\mathbf{x}_i, \boldsymbol{\theta}) = \prod_{i=1}^{M} \left( \left[ \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right] ^{y_i} \left[1- \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right] ^{1-y_i} \right)
\end{aligned}
$$

In order to do MLE, we first have log likelihood function:
$$
\begin{aligned}
\ln \left( \prod_{i=1}^{M} p(y_i|\mathbf{x}_i, \boldsymbol{\theta}) \right) = \sum_{i=1}^{M} \left({y_i} \ln \left( \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right)  +  (1-y_i) \ln \left(1- \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) \right) \right) 
\end{aligned}
$$

we may define hypothesis as
$$h(\boldsymbol{\theta}) = \hat{\mathbf{y}} = \sigma(\boldsymbol{\theta}^{\top} \mathbf{x}_i) = \sigma(\mathbf{z})$$

we define **cross-entropy** loss as the negative log-likelihood function, by minimizing the loss function, equivalently we get Max likelihood estimation.
$$
\begin{aligned}
L(\boldsymbol{\theta})
&= - \sum_{i=1}^{M} \left[ {\mathbf{y}_i} \ln \left( \hat{\mathbf{y}_i} \right)  +  (1-\mathbf{y}_i) \ln \left(1- \hat{\mathbf{y}_i} \right) \right] 
\end{aligned}
$$

The derivatives are computed as:
% Then we have,
% $$\frac{\partial h(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \frac{\partial {(1+e^{-\boldsymbol{\theta}^{\top}\mathbf{x}})}^{-1}} {\partial \boldsymbol{\theta}}  = h(\boldsymbol{\theta}) (1-h(\boldsymbol{\theta})) \mathbf{x} $$

First the derivative of loss function over prediction is, 
$$
\begin{aligned}
    \nabla_{\hat{\mathbf{y}}_i}L = \frac{\partial L } {\partial \hat{\mathbf{y}_i}} 
    &= - \sum_{i=1}^{M} \left( \frac{\mathbf{y}_i}{\hat{\mathbf{y}_i}} - \frac{1 - \mathbf{y}_i} { 1- \hat{\mathbf{y}}_i} \right)
\end{aligned}
$$

Then the derivative of $\sigma(\mathbf{z})$ over $\mathbf{z}$
$$\sigma'(\mathbf{z}) = \frac{e^{-\mathbf{z}}}{(1+e^{-\mathbf{z}})^2} = \frac{1}{(1+e^{-\mathbf{z}})} \frac{e^{-\mathbf{z}}}{(1+e^{-\mathbf{z}})} = \frac{1}{(1+e^{-\mathbf{z}})} \frac{1+e^{-\mathbf{z}}-1}{(1+e^{-\mathbf{z}})} = \sigma(\mathbf{z})(1-\sigma(\mathbf{z})) = \hat{\mathbf{y}} (1 - \hat{\mathbf{y}}) $$

Then the derivative of $\mathbf{z}$ over $\boldsymbol{\theta}$
$$\mathbf{z}' = \mathbf{x}$$

Then the derivative of loss over $\boldsymbol{\theta}$
$$
\begin{aligned}
    \nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta}) 
    &= - \sum_{i=1}^{M} \left( \frac{\mathbf{y}_i } {\hat{\mathbf{y}_i} } - \frac{1 - \mathbf{y}_i} { 1- \hat{\mathbf{y}}_i} \right) \hat{\mathbf{y}}_i (1 - \hat{\mathbf{y}}_i) \mathbf{x}_i \\
    &= - \sum_{i=1}^{M} \left(\mathbf{y}_i - \hat{\mathbf{y}}_i \right) \mathbf{x}_i = \sum_{i=1}^{M} \left(\hat{\mathbf{y}}_i - \mathbf{y}_i\right) \mathbf{x}_i \\
    & = \mathbf{X} (\mathbf{\hat{y} - \mathbf{y}})
\end{aligned}
$$

where $\left(\mathbf{\hat{y}} - \mathbf{y}_i \right)$ is the training error for the $i-$th sample.

\vspace{3mm}
Full gradient descent update:
$$
    \boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta})
$$


### Multi-class soft-max Logistic Regression
we will have a matrix $\mathbf{W} \in \mathbb{R}^{C \times d}$, where $C$ is the number of classes and $d$ is the number of dimension.

We have the loss function for $N$ samples, which means $\mathbf{X} \in \mathbb{R}^{d \times N}$. And for each entry $\mathbf{x}_n \in \mathbb{R}^d$, $\mathbf{y}_n \in \mathbb{R}^C$

$$
\begin{aligned}
L(\mathbf{W}) &= - \sum_{n=1}^N \mathbf{y}_n \ln p(\hat{\mathbf{y}}_n|\mathbf{x}_n) \\
&= - \sum_{n=1}^N \sum_{c=1}^C y_{nc} \ln p(c|\mathbf{x}_n) \\
&= - \sum_{n=1}^N \sum_{c=1}^C y_{nc} \ln \left( \frac{\exp{(\mathbf{w_c}^{\top}\mathbf{x_n}} ) } {\sum_{c'} \exp{(\mathbf{w_{c'}}^{\top} \mathbf{x_n}})} \right) \\
\end{aligned}
$$

Consider one class $c$ and one sample $\mathbf{x}_n = \mathbf{x}$
$$
\begin{aligned}
\ln p(c|\mathbf{x}) &= \ln \left( \frac{\exp{(\mathbf{w}^{\top}_c\mathbf{x}} ) } {\sum_{c'} \exp{(\mathbf{w}_{c'}^{\top} \mathbf{x}})} \right) \\
&= \ln ( e^{\mathbf{w}_c^{\top} \mathbf{x}})  - \ln \left( \sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} \right) \\
&= \mathbf{w}_c^{\top} \mathbf{x}  - \ln \left( \sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} \right)
\end{aligned}
$$

compute the gradient w.r.t $\mathbf{w}_c$
$$
\begin{aligned}
\frac{ \partial \ln p(c|\mathbf{x})} {\partial \mathbf{w}_c} 
&= \mathbf{x} - \frac{\partial  \left( \sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} \right) / \partial \mathbf{w}_c}{\sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} } \\
&= \mathbf{x} - \frac{ \mathbf{x} e^{\mathbf{w}^{\top}_c \mathbf{x}} }{\sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} } \\
&= \mathbf{x} \left( 1 - \frac{ e^{\mathbf{w}^{\top}_c \mathbf{x}} }{\sum_{c'} e^{\mathbf{w}_{c'}^{\top} \mathbf{x}} } \right) \\
&= \mathbf{x} \left( 1 - p(c|\mathbf{x}) \right) \\
\end{aligned}
$$

Jacobian matrix is
$$
\begin{aligned}
    \begin{bmatrix} 
     \mathbf{x} \left( 1 - p(c_1|\mathbf{x}) \right)  \\
    \vdots  \\
     \mathbf{x} \left( 1 - p(c_C|\mathbf{x}) \right)  \\
    \end{bmatrix}
\end{aligned}
$$


## Perceptron algorithm

## SVM

Support Vector Machine (SVM) is usually used for binary classification. We define positive class as +1 and negative class -1. The input $\mathbf{x} \in \mathbb{R}^D$, we define a hyper-plane $\mathbf{P} \in \mathbb{R}^{D-1}$which separate the data into two classes (positive and negative). 

Consider a function $f: \mathbb{}{R}^D \xrightarrow{} \mathbb{R}$, with input data point $\mathbf{x} \in \mathbb{R}^D$, then we have function $f$,
$$f(\mathbf{x}) := \mathbf{w}^{\top}\mathbf{x} + b $$

We define the hyperplane $\mathbf{P}$ as,
$$f(\mathbf{x})=0$$

Actually the weights vector $\mathbf{w}$, is perpendicular to the hyper-plane $\mathbf{P}$. Hence $\mathbf{w}$ is a normal vector of the hyper-plane. $\mathbf{w}$ is also called support vector. Now we prove that $\mathbf{w}$is perpendicular to the hyper-plane: Given two points $\mathbf{x}_1$ and $\mathbf{x}_2$ on the hyper-plane, which means $\mathbf{w}^{\top}\mathbf{x}_1 + b = 0$ and $\mathbf{w}^{\top}\mathbf{x}_2 + b = 0$. 

Then we have,
$$\mathbf{w}^{\top}(\mathbf{x}_1 -\mathbf{x}_2) = 0$$
thus $\mathbf{w}$ is perpendicular to $\mathbf{x}_1 -\mathbf{x}_2$. Hence $\mathbf{w}$ is perpendicular to to the hyperplane since $\mathbf{x}_1 -\mathbf{x}_2$ is parallel to the hyperplane.

Hinge loss:
$$\sum_{n=1}^{N}\max \{0,1-y_n(\langle \mathbf{w}, \mathbf{x}_n \rangle + b \}$$
