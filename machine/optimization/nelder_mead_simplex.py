from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Callable, List, Sequence, Union

import numpy as np

from .minimization_result import MinimizationExitCondition, MinimizationResult


def _initialize_error_values(
    vertices: List[np.ndarray], objective_function: Callable[[np.ndarray, int], float]
) -> List[float]:
    """Evaluate the objective function at each vertex to create a corresponding list of error values for each vertex"""

    return [objective_function(v, -1) for v in vertices]


@dataclass
class _ErrorProfile:
    highest_index: int = 0
    next_highest_index: int = 0
    lowest_index: int = 0


def _evaluate_simplex(error_values: List[float]) -> _ErrorProfile:
    """Examine all error values to determine the error profile"""

    error_profile = _ErrorProfile()
    if error_values[0] > error_values[1]:
        error_profile.highest_index = 0
        error_profile.next_highest_index = 1
    else:
        error_profile.highest_index = 1
        error_profile.next_highest_index = 0

    for index in range(len(error_values)):
        error_value = error_values[index]
        if error_value <= error_values[error_profile.lowest_index]:
            error_profile.lowest_index = index
        if error_value > error_values[error_profile.highest_index]:
            error_profile.next_highest_index = error_profile.highest_index
            error_profile.highest_index = index
        elif error_value > error_values[error_profile.next_highest_index] and index != error_profile.highest_index:
            error_profile.next_highest_index = index
    return error_profile


def _compute_centroid(vertices: List[np.ndarray], error_profile: _ErrorProfile) -> np.ndarray:
    """Compute the centroid of all points except the worst"""

    num_vertices = len(vertices)
    centroid = np.zeros(num_vertices - 1)
    for i in range(num_vertices):
        if i != error_profile.highest_index:
            centroid = np.add(centroid, vertices[i])
    return centroid * (1.0 / (num_vertices - 1))


def _try_to_scale_simplex(
    scale_factor: float,
    error_profile: _ErrorProfile,
    vertices: List[np.ndarray],
    error_values: List[float],
    objective_function: Callable[[np.ndarray, int], float],
    evaluation_count: int,
) -> float:
    """Test a scaling operation of the high point, and replace it if it is an improvement"""

    # find the centroid through which we will reflect
    centroid = _compute_centroid(vertices, error_profile)

    # define the vector from the centroid to the high point
    centroid_to_high_point = np.subtract(vertices[error_profile.highest_index], centroid)

    # scale and position the vector to determine the new trial point
    new_point = np.add(centroid_to_high_point * scale_factor, centroid)

    # evaluate the new point
    new_error_value = objective_function(new_point, evaluation_count)

    # if it's better, replace the old high point
    if new_error_value < error_values[error_profile.highest_index]:
        vertices[error_profile.highest_index] = new_point
        error_values[error_profile.highest_index] = new_error_value

    return new_error_value


def _shrink_simplex(
    error_profile: _ErrorProfile,
    vertices: List[np.ndarray],
    error_values: List[float],
    objective_function: Callable[[np.ndarray, int], float],
    evaluation_count: int,
) -> None:
    """Contract the simplex uniformly around the lowest point"""

    lowest_vertex = vertices[error_profile.lowest_index]
    for i in range(len(vertices)):
        if i != error_profile.lowest_index:
            vertices[i] = np.add(vertices[i], lowest_vertex) * 0.5
            error_values[i] = objective_function(vertices[i], evaluation_count)
            evaluation_count += 1


class NelderMeadSimplex:
    """Class implementing the Nelder-Mead simplex algorithm, used to find a minima when no gradient is available.

    Called fminsearch() in Matlab. A description of the algorithm can be found at
    http://se.mathworks.com/help/matlab/math/optimizing-nonlinear-functions.html#bsgpq6p-11
    or
    https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    """

    def __init__(self, convergence_tolerance: float, max_function_evaluations: int, scale: float) -> None:
        self.convergence_tolerance = convergence_tolerance
        self.max_function_evaluations = max_function_evaluations
        self.scale = scale

    def find_minimum(
        self, objective_function: Callable[[np.ndarray, int], float], initial_guess: Union[np.ndarray, Sequence[float]]
    ) -> MinimizationResult:
        """Finds the minimum of the objective function with an intial pertubation"""

        # create the initial simplex
        if not isinstance(initial_guess, np.ndarray):
            initial_guess = np.array(initial_guess)
        vertices = self._initialize_vertices(initial_guess)

        evaluation_count = 0
        exit_condition = MinimizationExitCondition.NONE

        error_values = _initialize_error_values(vertices, objective_function)

        # iterate until we converge, or complete our permitted number of iterations
        while True:
            error_profile = _evaluate_simplex(error_values)

            # see if the range in point heights is small enough to exit
            if self._has_converged(error_values):
                exit_condition = MinimizationExitCondition.CONVERGED
                break

            # attempt a reflection of the simplex
            reflection_point_value = _try_to_scale_simplex(
                -1.0, error_profile, vertices, error_values, objective_function, evaluation_count
            )
            evaluation_count += 1
            if reflection_point_value <= error_values[error_profile.lowest_index]:
                # it's better than the best point, so attempt an expansion of the simplex
                _try_to_scale_simplex(2.0, error_profile, vertices, error_values, objective_function, evaluation_count)
                evaluation_count += 1
            elif reflection_point_value >= error_values[error_profile.next_highest_index]:
                # it would be worse than the second best point, so attempt a contraction to look
                # for an intermediate point
                current_worst = error_values[error_profile.highest_index]
                contraction_point_value = _try_to_scale_simplex(
                    0.5, error_profile, vertices, error_values, objective_function, evaluation_count
                )
                evaluation_count += 1
                if contraction_point_value >= current_worst:
                    # that would be even worse, so let's try to contract uniformly towards the low point;
                    # don't bother to update the error profile, we'll do it at the start of the
                    # next iteration
                    _shrink_simplex(error_profile, vertices, error_values, objective_function, evaluation_count)
                    # that required one function evaluation for each vertex; keep track
                    evaluation_count += len(vertices) - 1
            # check to see if we have exceeded our alloted number of evaluations
            if evaluation_count >= self.max_function_evaluations:
                exit_condition = MinimizationExitCondition.MAX_FUNCTION_EVALUATIONS
                break

        return MinimizationResult(
            exit_condition,
            vertices[error_profile.lowest_index],
            error_values[error_profile.lowest_index],
            evaluation_count,
        )

    def _initialize_vertices(self, initial_guess: np.ndarray) -> List[np.ndarray]:
        """Construct an initial simplex, given starting guesses for the constants,
        and initial step sizes for each dimension"""

        num_dimensions = initial_guess.shape[0]

        pn = self.scale * (sqrt(num_dimensions + 1) - 1 + num_dimensions) / (num_dimensions * sqrt(2))
        qn = self.scale * (sqrt(num_dimensions + 1) - 1) / (num_dimensions * sqrt(2))

        vertices: List[np.ndarray] = []
        p0 = initial_guess
        vertices.append(p0)
        for i in range(num_dimensions):
            v: List[float] = []
            for j in range(num_dimensions):
                v.append(pn if i == j else qn)
            vertices.append(np.add(p0, v))
        return vertices

    def _has_converged(self, error_values: List[float]) -> bool:
        """Check whether the points in the error profile have so little range that we
        consider ourselves to have converged"""

        avg = mean(error_values)
        r = 0
        for ev in error_values:
            r += pow(ev - avg, 2) / (len(error_values) - 1)
        r = sqrt(r)
        return r < self.convergence_tolerance
