# Bayesian Linear Regression method

To extend the understanding of the [electricity demanding project](https://github.com/natwonglakhon/QLD_Electricity_Forecast), I'd to see the derivation myself.

Let us consider the a simple modle:
$y_t = \vec X_t^\top \cdot \vec\beta_t + \epsilon,$

where $y_t$ is the traget values, $\vec X_t = (x_1, \dots, x_n)$ is the observed features (with $n$ elements), $\vec \beta_t = (\beta_1, \dots, \beta_n)$ is an unknown random vector (we want to estimate), and $\epsilon$ is an unknown random variables.

There are two assumptions made in this model:
 * $\epsilon \sim \mathcal{N}(0, \sigma)$. Meaning $\epsilon$ flucuates *normally* around zero with the variance $\sigma^2$. I'd like to think it is a bias of the model. (Of course, a good model should have zero bisas.)
 * $\vec \beta_t \sim \mathcal{N}(\mu_t, \Sigma_t)$. Meaning $\vec \beta_t$ is a normal random vector with $\vec \mu_t$ mean and $\vec \Sigma_t^2$ variance.

 **The question is how do we update** $\vec \beta_t$?

 Answer, use Baye's rule.

**The prior**: From the second assumption, we may write the distribution of $\vec \beta_t$ as:
$p(\vec\beta_t) \propto \exp[-\frac{1}{2}(\vec\beta_t - \vec\mu_t)^\top \vec \Sigma_t^{-1}(\vec \beta_t-\vec\mu_t)],$
where I have ignored the normalisation by leaving it with the proportional sign.


As $\epsilon$ has zero mean and $\sigma^2$ variance, it follows that $y_t|\vec X_t, \vec\beta_t$ has a mean of $\vec X_t^\top \cdot \vec\beta_t$ with $\sigma^2$ variance. Thus, we shall write:
$p(y_t|\vec\beta_t) \propto \exp[-\frac{1}{2}(y_t- \vec X_t^\top \cdot \vec\beta_t)^\top(y_t- \vec X_t^\top \cdot \vec\beta_t)/\sigma^2].$

**The posterior**: Now, we ask what is the probablity of $\vec\beta_t$ given the traget $y_t$?

To answer that, we use Baye's rule. Baye's rule states that $p(\vec\beta_t|y_t) \propto p(y_t|\vec\beta_t)p(\vec\beta_t)$. Therefore, we can use the expression above, giving:
$p(\vec\beta_t|y_t) \propto \exp[-\frac{1}{2}(y_t- \vec X_t^\top \cdot \vec\beta_t)^\top(y_t- \vec X_t^\top \cdot \vec\beta_t)/\sigma^2 - \frac{1}{2}(\vec\beta_t - \vec\mu_t)^\top \vec \Sigma_t^{-1}(\vec \beta_t-\vec\mu_t)].$
Note that it is just an expotential of a scalr so we can combine the argument together.

We can rearrage the expression above in the following form:
$p(\vec\beta_t|y_t) \propto \exp[-\frac{1}{2}(\vec\beta_t-\vec\mu_{t'})^\top \vec \Sigma_{t'} (\vec\beta_t-\vec\mu_{t'})],$
where we agian ignore the constant (unrelated to $\vec\beta_t$) and find that

 * $\vec\Sigma_{t'} = (\vec\Sigma_t^{-1}+ \frac{\vec X_t^\top\vec X_t}{\sigma^2})^{-1}$ and
 * $\vec\mu_{t'}=\vec\Sigma_{t'}(\vec\Sigma_t^{-1}\vec\mu_t + \frac{\vec X_t^T y_t}{\sigma^2})$.

 The two expressions above give the recusive update for the parameter $\vec\beta_{t'}$.


 ## Implement the model using BayesianRidge

 From sklearn, BayesianRidge is available to use. However, BayesianRidge does *Empirical Bayes* which is slightly different from the thoery. I will not go to detials for now.