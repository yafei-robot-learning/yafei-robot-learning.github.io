# Models for Classification

## Logistic regression
The input data (or features) is denoted as $\mathbf{X} \in \mathbb{R} ^{d \times N}$, and its labels are $\mathbf{y} \in \mathbb{R} ^{N \times 1}$. $N$ is the size of the dataset, and $d$ is the feature dimension. The weight of the regression model is $\mathbf{w} \in \mathbb{R}^{d \times 1}$.

In matrix form:
$$
\mathbf{X} =
\begin{bmatrix}
| & | &        & | \\
\mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_N \\
| & | &        & |
\end{bmatrix}
\in \mathbb{R}^{d \times N},

\quad
\mathbf{y} =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix}
\in \mathbb{R}^{N \times 1},

\quad
\mathbf{w} =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_d
\end{bmatrix}
\in \mathbb{R}^{d \times 1}
$$

The probability of the prediction being 1 is,
$$p(y=1|\mathbf{x}, \mathbf{w}) = \sigma(\mathbf{w}^{\top} \mathbf{x})$$

where $\sigma(.)$ is the sigmoid function. $$\sigma(x) = \frac{1}{1+e^{-x}}$$

![Sigmoid function logistic curve](../figs/sigmoid.png "width=50%")

Sigmoid function properties:
- Bounded between 0 and 1, thus can be interpreted as probability
- Monotonically increasing, thus usable to derive classification rules, e.g.
  - $\sigma(x) \geq 0.5$, positive (classify as ’1’)
  - $\sigma(x) < 0.5$, negative (classify as ’0’)

Next we derive the loss function.
For a single sample $(\mathbf{x}_i, y_i)$, where $\mathbf{x}, \mathbf{w} \in \mathbb{R} ^{d \times 1}$

$$p(y_i|\mathbf{x}_i, \mathbf{w}) = \begin{cases}
      \sigma(\mathbf{w}^{\top} \mathbf{x}_i) & \text{if $y_i=1$ }\\
      1-\sigma(\mathbf{w}^{\top} \mathbf{x}_i) & \text{otherwise}
    \end{cases}  
$$

since $y_i$ is either 0 or 1, we can rewrite this as
$$p(y_i|\mathbf{x}_i, \mathbf{w}) = \left[ \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right] ^{y_i} \left[1- \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right] ^{1-y_i}$$

extend to whole dataset we define energy function which is the total likelihood:
$$
\begin{aligned}
E = \prod_{i=1}^{N} p(y_i|\mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^{N} \left( \left[ \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right] ^{y_i} \left[1- \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right] ^{1-y_i} \right)
\end{aligned}
$$

In order to do MLE, we first have log likelihood function:
$$
\begin{aligned}
\ln \left( \prod_{i=1}^{N} p(y_i|\mathbf{x}_i, \mathbf{w}) \right) = \sum_{i=1}^{N} \left({y_i} \ln \left( \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right)  +  (1-y_i) \ln \left(1- \sigma(\mathbf{w}^{\top} \mathbf{x}_i) \right) \right) 
\end{aligned}
$$

we may define hypothesis as
$$h(\mathbf{w}) = \hat{\mathbf{y}} = \sigma(\mathbf{w}^{\top} \mathbf{x}_i) = \sigma(\mathbf{z})$$

we define **cross-entropy** loss as the negative log-likelihood function, by minimizing the loss function, equivalently we get Max likelihood estimation.
$$
\begin{aligned}
L(\mathbf{w})
&= - \sum_{i=1}^{N} \left[ {\mathbf{y}_i} \ln \left( \hat{\mathbf{y}_i} \right)  +  (1-\mathbf{y}_i) \ln \left(1- \hat{\mathbf{y}_i} \right) \right] 
\end{aligned}
$$

The derivatives are computed as:
% Then we have,
% $$\frac{\partial h(\mathbf{w})}{\partial \mathbf{w}} = \frac{\partial {(1+e^{-\mathbf{w}^{\top}\mathbf{x}})}^{-1}} {\partial \mathbf{w}}  = h(\mathbf{w}) (1-h(\mathbf{w})) \mathbf{x} $$

First the derivative of loss function over prediction is, 
$$
\begin{aligned}
    \nabla_{\hat{\mathbf{y}}_i}L = \frac{\partial L } {\partial \hat{\mathbf{y}_i}} 
    &= - \sum_{i=1}^{N} \left( \frac{\mathbf{y}_i}{\hat{\mathbf{y}_i}} - \frac{1 - \mathbf{y}_i} { 1- \hat{\mathbf{y}}_i} \right)
\end{aligned}
$$

Then the derivative of $\sigma(\mathbf{z})$ over $\mathbf{z}$
$$\sigma'(\mathbf{z}) = \frac{e^{-\mathbf{z}}}{(1+e^{-\mathbf{z}})^2} = \frac{1}{(1+e^{-\mathbf{z}})} \frac{e^{-\mathbf{z}}}{(1+e^{-\mathbf{z}})} = \frac{1}{(1+e^{-\mathbf{z}})} \frac{1+e^{-\mathbf{z}}-1}{(1+e^{-\mathbf{z}})} = \sigma(\mathbf{z})(1-\sigma(\mathbf{z})) = \hat{\mathbf{y}} (1 - \hat{\mathbf{y}}) $$

Then the derivative of $\mathbf{z}$ over $\mathbf{w}$
$$\mathbf{z}' = \mathbf{x}$$

Then the derivative of loss over $\mathbf{w}$
$$
\begin{aligned}
    \nabla_{\mathbf{w}}L(\mathbf{w}) 
    &= - \sum_{i=1}^{N} \left( \frac{\mathbf{y}_i } {\hat{\mathbf{y}_i} } - \frac{1 - \mathbf{y}_i} { 1- \hat{\mathbf{y}}_i} \right) \hat{\mathbf{y}}_i (1 - \hat{\mathbf{y}}_i) \mathbf{x}_i \\
    &= - \sum_{i=1}^{N} \left(\mathbf{y}_i - \hat{\mathbf{y}}_i \right) \mathbf{x}_i = \sum_{i=1}^{N} \left(\hat{\mathbf{y}}_i - \mathbf{y}_i\right) \mathbf{x}_i \\
    & = \mathbf{X} (\mathbf{\hat{y} - \mathbf{y}})
\end{aligned}
$$

where $\left(\mathbf{\hat{y}} - \mathbf{y}_i \right)$ is the training error for the $i-$th sample.

Full gradient descent update:
$$
    \mathbf{w}_{t+1} \leftarrow \mathbf{w}_t - \eta \nabla_{\mathbf{w}}L(\mathbf{w})
$$


### Multi-class soft-max Logistic Regression
we will have a matrix $\mathbf{W} \in \mathbb{R}^{C \times d}$, where $C$ is the number of classes and $d$ is the number of dimension.

For $N$ samples, $\mathbf{X} \in \mathbb{R}^{d \times N}$ and $\mathbf{Y} \in \mathbb{R}^{C \times N}$. For each sample, $\mathbf{x}_n \in \mathbb{R}^{d \times 1}$ and $\mathbf{y}_n \in \mathbb{R}^{C \times 1}$.

The matrix form is:
$$
\begin{aligned}
\mathbf{Z} &= \mathbf{W}\mathbf{X} \in \mathbb{R}^{C \times N} \\
\hat{\mathbf{Y}} &= \text{softmax}(\mathbf{Z}) = \text{softmax}(\mathbf{W}\mathbf{X}) \in \mathbb{R}^{C \times N}
\end{aligned}
$$

where the softmax is applied column-wise:
$$
\hat{y}_{cn}
= p(c|\mathbf{x}_n)
= \frac{\exp(\mathbf{w}_c^{\top}\mathbf{x}_n)}
{\sum_{c'=1}^{C}\exp(\mathbf{w}_{c'}^{\top}\mathbf{x}_n)}
$$

This comes from the likelihood of one sample:
$$
p(\mathbf{y}_n|\mathbf{x}_n, \mathbf{W})
= \prod_{c=1}^{C} p(c|\mathbf{x}_n)^{y_{cn}}
$$

For the whole dataset:
$$
\begin{aligned}
p(\mathbf{Y}|\mathbf{X}, \mathbf{W})
&= \prod_{n=1}^{N} \prod_{c=1}^{C} p(c|\mathbf{x}_n)^{y_{cn}} \\
\ln p(\mathbf{Y}|\mathbf{X}, \mathbf{W})
&= \sum_{n=1}^{N} \sum_{c=1}^{C} y_{cn}\ln p(c|\mathbf{x}_n)
\end{aligned}
$$

Therefore, minimizing the negative log-likelihood gives:
$$
\begin{aligned}
L(\mathbf{W})
&= - \sum_{n=1}^N \sum_{c=1}^C y_{cn} \ln p(c|\mathbf{x}_n) \\
&= - \sum_{n=1}^N \mathbf{y}_n^{\top}\ln \hat{\mathbf{y}}_n \\
&= - \langle \mathbf{Y}, \ln \hat{\mathbf{Y}} \rangle_F \\
&= - \mathrm{tr} \left(\mathbf{Y}^{\top}\ln \hat{\mathbf{Y}}\right) \\
\end{aligned}
$$

where $\langle . \rangle_F$ is the Frobenius inner product.

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

The derivation above is only for $\mathbf{w}_c$. For any class weight $\mathbf{w}_k$,
$$
\begin{aligned}
\frac{\partial \ln p(c|\mathbf{x})}{\partial \mathbf{w}_k}
&= \left(\mathbb{1}_{k=c} - p(k|\mathbf{x}) \right)\mathbf{x}
\end{aligned}
$$

For one sample, the cross-entropy loss is:
$$
\begin{aligned}
\ell(\mathbf{W})
&= - \sum_{c=1}^{C} y_c \ln p(c|\mathbf{x})
\end{aligned}
$$

Then the gradient of the loss with respect to $\mathbf{w}_k$ is:
$$
\begin{aligned}
\frac{\partial \ell(\mathbf{W})}{\partial \mathbf{w}_k}
&= - \sum_{c=1}^{C} y_c
\frac{\partial \ln p(c|\mathbf{x})}{\partial \mathbf{w}_k} \\
&= - \sum_{c=1}^{C} y_c
\left(\mathbb{1}_{k=c} - p(k|\mathbf{x}) \right)\mathbf{x} \\
&= \left(p(k|\mathbf{x}) - y_k \right)\mathbf{x}
\end{aligned}
$$

For the whole dataset:
$$
\begin{aligned}
\frac{\partial L(\mathbf{W})}{\partial \mathbf{w}_k}
&= \sum_{n=1}^{N} \left(p(k|\mathbf{x}_n) - y_{kn} \right)\mathbf{x}_n
\end{aligned}
$$

In matrix form:
$$
\begin{aligned}
\nabla_{\mathbf{W}} L(\mathbf{W})
&= (\hat{\mathbf{Y}} - \mathbf{Y})\mathbf{X}^{\top}
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
