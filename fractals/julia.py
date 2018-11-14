"""Creates a matrix whose entries represent complex numbers on a grid and
whose values represent the escaping time for each number. The escaping time is
defined as follows:

    Let
        f(z) = (z^2 + c)/(z^2 - c)
    where c is 0.7 for example. Then the escaping time i for a given z is
        inf {i in Z s.t. |f^(i)(z)| > 2}
    where for example
        f^(3)(z) = f(f(f(z)))

We then plot the matrix to get the fractal.
"""

import numpy as np
import matplotlib.pyplot as plt
import click


@click.command()
@click.option('--c', default=0.7, help="The c value in the iterative function.")
@click.option('--max-iter', default=100, help="Maximum iterations of the function before we assume values have not escaped i.e. are oscillating.")
@click.option('--escape-value', default=2, help="Value to reach at which point we assume it has escaped i.e. not oscillating.")
@click.option('--x-max', default=1, help="Grid boundaries on real numbers are [-x-max, x-max].")
@click.option('--x-step', default=0.01, help="Grid boundaries on real numbers have step size x-step.")
@click.option('--y-max', default=1, help="Grid boundaries on complex numbers are [-y-max, y-max].")
@click.option('--y-step', default=0.01, help="Grid boundaries on complex numbers have step size y-step.")
def julia(c, max_iter, escape_value, x_max, x_step, y_max, y_step):
    """Plots a Julia set fractal."""
    # Initialise matrix for representing the f^(i)(z).
    x_points = np.arange(-x_max, x_max+x_step, x_step, dtype=np.complex128)
    y_points = np.arange(-y_max, y_max+y_step, y_step, dtype=np.complex128) * 1j

    W = x_points[:, None] + y_points[None, :]

    # Initialise matrix for representing escaping time.
    E = -np.ones(W.shape, dtype=np.int32)

    # Define function f that we can apply to W.
    def f(W):
        return (W**2 + c)/(W**2 - c)

    # Calculate the escaping times.
    for i in range(max_iter):
        # Find escaped values.
        idx = (np.absolute(W) > escape_value) & (E == -1)
        E[idx] = i

        # Apply function to W.
        W = f(W)


    # Plot the image.
    plt.imshow(E)
    plt.show()


if __name__ == "__main__":
    julia()
