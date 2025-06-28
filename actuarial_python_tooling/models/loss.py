import jax
import jax.numpy as jnp

# https://github.com/dmlc/xgboost/blob/7663de956c37eb4dd528132214e68ba2851d9696/src/metric/elementwise_metric.cu#L270-L286
# https://www.math.cit.tum.de/fileadmin/w00ccg/math/Forschung/forschungsgruppen/statistics/academics/lec8.pdf

# l(mu, v, y) = (v - 1) * log(y)- (v/mu) * y+ vlog(v) - vlog(y) - log(Gamma(v))
#

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_loss/loss.py#L748

#        loss(x_i) = log(exp(raw_prediction_i)/y_true_i)
#                    + y_true/exp(raw_prediction_i) - 1

# Gamma Deviance:
# https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/metrics/_regression.py#L1356

def gamme_neg_log_loss(shape, scale, y, weights) -> jax.Array:
    res = 


def gamma_neg_log_loss(y_pred, X, y, weights) -> jax.Array:
    res = jnp.log(y_pred / y) + (y / y_pred) - 1
    return res


@jax.jit
def poisson_neg_log_loss(beta, X, y, weights) -> jax.Array:
    """
    Returns the negative poisson log likelihood as implemented in: https://jax.quantecon.org/mle.html

    The only change is the use of the mean vs. the sum. This should make it a lot easier to use the
    loss function inside of any batched update procedure (ie. sgd). Since the loss will not directly
    depend on the batch size.

    The factorial is actually important for finding the correct coefficients! Without it the weighted mean of the predictions
    will be correct, but the model will still be bad if one checks a subset of predictions via a simple lift plot for instance.

    PoissonNLLLoss in PyTorch also includes that part of the loss: https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss
    """
    μ = jnp.exp(X @ beta)
    if weights is not None:
        return -1 * jnp.mean((y * jnp.log(μ) - μ - jnp.log(jax_factorial(y))) * weights)
    else:
        return -1 * jnp.mean((y * jnp.log(μ) - μ - jnp.log(jax_factorial(y))))
