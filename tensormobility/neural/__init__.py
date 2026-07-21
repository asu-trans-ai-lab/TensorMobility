"""tensormobility.neural: the neural <-> transportation identity,
executable. See docs/TENSOR_AXES.md."""
from tensormobility.neural.correspondence import (
    softmax_router, logit_choice, router_is_logit,
    routed_reduced_cost, hard_routing_limit, admission_complementarity)
from tensormobility.neural.tcg_graph import (
    masked_softmax_rows, forward, loss_and_grad, fd_check)
