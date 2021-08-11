import re

from . import libqapopt as lib
from .model import Model


def parse_dd_model(f):
    re_p = re.compile(r'^p ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)$')
    re_a = re.compile(r'^a ([0-9]+) ([0-9]+) ([0-9]+) (.+)$')
    re_e = re.compile(r'^e ([0-9]+) ([0-9]+) (.+)$')
    re_d = re.compile(r'^d ([0-9]+) ([0-9]+) ([0-9]+) (.+)$')
    re_ignore = re.compile(r'^(c|i0|i1|n0|n1) ')

    for line in f:
        line = line.rstrip()

        m = re_p.search(line)
        if m:
            model = Model(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            continue

        m = re_a.search(line)
        if m:
            model.add_assignment(int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))
            continue

        m = re_d.search(line)
        if m:
            model.add_dummy(int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))
            continue

        m = re_e.search(line)
        if m:
            model.add_edge(int(m.group(1)), int(m.group(2)), float(m.group(3)))
            continue

        m = re_ignore.search(line)
        if not m:
            assert False, "unknown input line"

    assert model.no_assignments == len(model.assignments)
    assert model.no_edges == len(model.edges)

    return model


def parse_sol(f):
    sol = []
    for line in f:
        line = line.rstrip()
        line = line.split()
        if line != []:
            assert int(line[0]) == len(sol)
            sol.append(int(line[1]))
    return sol
