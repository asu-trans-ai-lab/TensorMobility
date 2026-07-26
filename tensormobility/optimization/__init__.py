"""tensormobility.optimization: joint optimization structure over the
integrated column space.

  quadratic_pricing      gradient / Hessian decomposition of the
                         integrated objective and second-order
                         finite-move column pricing
  mobility_tensor_case   the typed grid activity-tour paper case
                         (complete columns, U_ijm opportunity field,
                         mobility-tensor projections)
"""
from tensormobility.optimization.quadratic_pricing import (
    ReplacementPrice,
    exact_finite_change,
    integrated_gradient,
    integrated_hessian,
    quadratic_price,
    replacement_direction,
    third_order_error_bound,
)

__all__ = [
    "ReplacementPrice",
    "exact_finite_change",
    "integrated_gradient",
    "integrated_hessian",
    "quadratic_price",
    "replacement_direction",
    "third_order_error_bound",
]
