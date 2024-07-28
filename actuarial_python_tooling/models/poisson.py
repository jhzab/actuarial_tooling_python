from functools import partial
import jax
import jax.numpy as jnp
import jaxopt
import optax


"""
Important:
Data needs to be properly formatted to avoid problem in loss function if loss aggregates via sum.
Even in that case problems with +/-inf might appear if log/exp is involved!
"""


@jax.jit
def poisson_neg_log_loss(beta, X, y, weights) -> jax.Array:
    μ = jnp.exp(X @ beta)
    if weights is not None:
        return -1 * jnp.mean(
            (y * jnp.log(μ) - μ) * weights
        )  # mean vs sum: https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641/2
    else:
        return -1 * jnp.mean(
            (y * jnp.log(μ) - μ)
        )  # mean vs sum: https://discuss.pytorch.org/t/loss-reduction-sum-vs-mean-when-to-use-each/115641/2


def poisson_neg_log_loss_no_jit(beta, X, y) -> jax.Array:
    μ = jnp.exp(X @ beta)
    print(X @ beta)
    print(μ.flatten())
    return -1 * jnp.sum(y * jnp.log(μ) - μ)


poisson_neg_log_loss_gradient = jax.grad(poisson_neg_log_loss)
poisson_neg_log_loss_hessian = jax.jacfwd(poisson_neg_log_loss_gradient)


def newton_raphson(X, y, beta, loss, gradient, hessian, tol=1e-3, max_iter=100, display=True):
    i = 0
    error = 100  # Initial error value

    # Print header of output
    if display:
        header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"θ":<60}'
        print(header)
        print("-" * len(header))

    # While loop runs while any value in error is greater
    # than the tolerance until max iterations are reached
    update = f"{-1:<13}{loss(X, y, beta):<16.8}{beta}"
    print(update)
    while jnp.any(error > tol) and i < max_iter:
        H, G = jnp.squeeze(hessian(X, y, beta)), gradient(X, y, beta)
        beta_new = beta - (jnp.dot(jnp.linalg.inv(H), G))
        error = jnp.abs(beta_new - beta)
        beta = beta_new

        if display:
            beta_list = [f"{t:.3}" for t in list(β.flatten())]
            update = f"{i:<13}{loss(X, y, beta):<16.8}{beta_list}"
            print(update)

        i += 1

    print(f"Number of iterations: {i}")
    print(f"beta_hat = {beta.flatten()}")

    return beta


def solve_via_jaxopt(X, y, weights, init_β):
    init_β = init_β.reshape(X.shape[1], 1)

    # Create an object with Poisson model values
    _loss = partial(poisson_neg_log_loss, X=X, y=y, weights=weights)

    solver = jaxopt.LBFGS(fun=_loss, maxiter=5)
    return solver.run(init_β)
    # res = jminimize(poisson_logL_p, init_β, method="BFGS")  # hess=...


def solve_via_jaxopt_scipy(X, y, weights, init_β):
    init_β = init_β.reshape(X.shape[1], 1)
    _loss = partial(poisson_neg_log_loss, X=X, y=y, weights=weights)
    solver = jaxopt.ScipyMinimize(fun=_loss, maxiter=20, method="L-BFGS-B")
    return solver.run(init_β)


def solve_via_optax(X, y, weights, init_β):
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(init_β)

    value_and_grad = optax.value_and_grad_from_state(poisson_neg_log_loss)
    for _ in range(10):
        val, grad = value_and_grad(init_β, X, y, weights, state=opt_state)
        print("value (objective function)", val)
        updates, opt_state = optimizer.update(
            grad, opt_state, init_β, value=val, grad=grad, value_fn=poisson_neg_log_loss, X=X, y=y, weights=weights
        )
        init_β = optax.apply_updates(init_β, updates)

    return init_β


def solve_via_adam(X, y, weights, init_β):
    init_β = init_β.reshape(X.shape[1], 1)

    # Use newton_raphson to find the MLE
    # optimizer = optax.chain(optax.lbfgs(learning_rate=0.0002), linesearch)
    optimizer = optax.adam(learning_rate=0.01, b1=0.4, b2=0.5)

    # Initialize parameters of the model + optimizer.
    opt_state = optimizer.init(init_β)

    print("Objective function: ", poisson_neg_log_loss(init_β, X, y, weights))
    for _ in range(1000):
        grad = jax.grad(poisson_neg_log_loss)(init_β, X, y, weights)
        updates, opt_state = optimizer.update(grad, opt_state)
        init_β = optax.apply_updates(init_β, updates)

    print("beta", init_β)
    print("Objective function: ", poisson_neg_log_loss(init_β, X, y, weights))

    return init_β.flatten()
