# Bayesian Linear Regression method
To extend the understanding of the [electricity demand project](https://github.com/natwonglakhon/QLD_Electricity_Forecast), I'd like to see the derivation myself.
Let us consider a simple model:

$$y_t = \vec X_t^\top \cdot \vec\beta_t + \epsilon,$$

where $y_t$ is the target values, $\vec X_t = (x_1, \dots, x_n)$ is the observed features (with $n$ elements), $\vec \beta_t = (\beta_1, \dots, \beta_n)$ is an unknown random vector (we want to estimate), and $\epsilon$ is an unknown random variable.

There are two assumptions made in this model:
 * $\epsilon \sim \mathcal{N}(0, \sigma)$. Meaning $\epsilon$ fluctuates *normally* around zero with the variance $\sigma^2$. I'd like to think it is a bias of the model. (Of course, a good model should have zero bias.)
 * $\vec \beta_t \sim \mathcal{N}(\vec\mu_t, \hat\Sigma_t)$. Meaning $\vec \beta_t$ is a normal random vector with $\vec \mu_t = (\mu_1, \dots, \mu_n)$ mean and $\hat \Sigma_t^2$ covariance matrix with $$\hat \Sigma_t = \begin{pmatrix} 
\Sigma_{11} & \dots & \Sigma_{1n}\\ 
\vdots & \ddots & \vdots\\ 
\Sigma_{n1} & \dots & \Sigma_{nn} 
\end{pmatrix}.$$

**The question is how do we update** $\vec \beta_t$?

Answer, use Bayes' rule.

**The prior**: From the second assumption, we may write the distribution of $\vec \beta_t$ as:

$$p(\vec\beta_t) \propto \exp\left[-\frac{1}{2}(\vec\beta_t - \vec\mu_t)^\top \hat\Sigma_t^{-1}(\vec \beta_t-\vec\mu_t)\right],$$

where I have ignored the normalisation by leaving it with the proportional sign.

As $\epsilon$ has zero mean and $\sigma^2$ variance, it follows that $y_t\|\vec X_t, \vec\beta_t$ has a mean of $\vec X_t^\top \cdot \vec\beta_t$ with $\sigma^2$ variance. Thus, we shall write:

$$p(y_t \mid \vec\beta_t) \propto \exp\left[-\frac{1}{2}(y_t- \vec X_t^\top \cdot \vec\beta_t)^\top(y_t- \vec X_t^\top \cdot \vec\beta_t)/\sigma^2\right].$$

**The posterior**: Now, we ask what is the probability of $\vec\beta_t$ given the target $y_t$?

To answer that, we use Bayes' rule. Bayes' rule states that $p(\vec\beta_t\|y_t) \propto p(y_t\|\vec\beta_t)p(\vec\beta_t)$. Therefore, we can use the expression above, giving:

$$p(\vec\beta_t \mid y_t) \propto \exp\left[-\frac{1}{2}(y_t- \vec X_t^\top \cdot \vec\beta_t)^\top(y_t- \vec X_t^\top \cdot \vec\beta_t)/\sigma^2 - \frac{1}{2}(\vec\beta_t - \vec\mu_t)^\top \hat\Sigma_t^{-1}(\vec \beta_t-\vec\mu_t)\right].$$

Note that it is just an exponential of a scalar so we can combine the argument together.

We can rearrange the expression above in the following form (via complete square):

$$p(\vec\beta_t \mid y_t) \propto \exp\left[-\frac{1}{2}(\vec\beta_t-\vec\mu_{t'})^\top \vec \Sigma_{t'} (\vec\beta_t-\vec\mu_{t'})\right],$$

where we again ignore the constant (unrelated to $\vec\beta_t$). Note that the above expression can be achieve by using the symetric property of $\hat\Sigma_t^{-1}$. We find the updated mean and variance as
 * $\vec\Sigma_{t'} = (\hat\Sigma_t^{-1}+ \frac{\vec X_t^\top\cdot\vec X_t}{\sigma^2})^{-1}$ and
 * $\vec\mu_{t'}=\vec\Sigma_{t'}(\hat\Sigma_t^{-1}\cdot\vec\mu_t + \frac{\vec X_t^T y_t}{\sigma^2})$.

The two expressions above give the recursive update for the parameter $\vec\beta_{t'}$. The time incoming observed features will be fed and the vector $\vec\beta$ will be updated recursively. 

## Implement the model using BayesianRidge
From sklearn, BayesianRidge is available to use. However, BayesianRidge does *Empirical Bayes* which is slightly different from the theory. I will not go into details for now.