from . import config

if config.ENABLE_GUROBI:
    from gurobipy import *
else:
    GRB = ()

    class Env:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('ENABLE_GUROBI not enabled during configuration of libqapopt.')

    class Model:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('ENABLE_GUROBI not enabled during configuration of libqapopt.')
