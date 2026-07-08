# Bayesian Linear Regression method

To extend the understanding of the [electricity demanding project](https://github.com/natwonglakhon/QLD_Electricity_Forecast), I'd to see the derivation myself.

Let us consider the a simple modle:
$$ y_t = \hat X_t^\top \cdot \beta_t + \epsilon,$$

where $y_t$ is the traget values, $\vec X_t = (x_1, \dots, x_n)$ is the observed features (with $n$ elements), $\vec \beta_t = (\beta_1, \dots, \beta_n)$ is an unknown random vector (we want to estimate), and $\epsilon$ is an unknown random variables.

There are two assumptions made in this model:
 - $\epsilon \sim \mathcal{N}(0, \sigma)$. Meaning $\epsilon$ flucuates *normally* around zero with the variance $\sigma^2$. I'd like to think it is a bias of the model. (Of course, a good model should have zero bisas.)
 - $\vec \beta_t \sim \mathcal{N}(\mu_t, \Sigma_t)$. Meaning $\vec \beta_t$ is a normal random vector with $\vec \mu_t$ mean and $\vec \Sigma_t^2$ variance.

 ## The question is how do we update $\vec \beta_t$?

 Answer, use Baye's rule.

**The prior**: From the second assumption, we may write the distribution of $\vec \beta_t$ as:
$$p(\vec\beta_t) = \exp[-\frac{1}{2}(\vec\beta_t - \vec\mu_t)^\top \vec \Sigma_t^{-1}(\vec \beta_t-\vec\mu_t)].$$
