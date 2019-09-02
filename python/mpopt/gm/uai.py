import numpy

from .model import Model


def tokenize_file(f):
    import re
    r = re.compile(r'[a-zA-Z0-9.+-]+')
    for line in f:
        line = line.rstrip('\r\n')
        for m in r.finditer(line):
            yield m.group(0)


def parse_uai(tokens):
    header = next(tokens)
    assert header == 'MARKOV'
    model = Model()

    num_nodes = int(next(tokens))
    for i in range(num_nodes):
        idx = model.add_unary(numpy.zeros(int(next(tokens)), dtype=numpy.float64))

    node_list = []
    num_costs = int(next(tokens))
    for i in range(num_costs):
        num_vars = int(next(tokens))
        node_list.append(tuple(int(next(tokens)) for j in range(num_vars)))

    for i in range(num_costs):
        size = int(next(tokens))
        if len(node_list[i]) == 1:
            u, = node_list[i]
            data = model.unaries[u]
            assert size == len(data)
            for x_u in range(len(data)):
                data[x_u] = float(next(tokens))
        elif len(node_list[i]) == 2:
            u, v = node_list[i]
            no_u = len(model.unaries[u])
            no_v = len(model.unaries[v])
            assert size == no_u * no_v
            data = numpy.zeros((no_u, no_v), dtype=numpy.float64)
            for x_u in range(no_u):
                for x_v in range(no_v):
                    data[x_u, x_v] = float(next(tokens))
            model.add_pairwise(u, v, data)
        else:
            raise RuntimeError('Higher-order factors not supported.')

    return model


def parse_uai_file(f):
    return parse_uai(tokenize_file(f))


def write_uai_file(model, f):
    f.write('MARKOV\n')
    f.write('{}\n'.format(len(model.unaries)))

    f.write('{}'.format(' '.join(str(len(x)) for x in model.unaries)))
    f.write('\n')

    f.write('{}'.format(len(model.unaries) + len(model.pairwise)))
    f.write('\n')

    for u, _ in enumerate(model.unaries):
        f.write('1 {}\n'.format(u))

    for left, right, _ in model.pairwise:
        f.write('2 {} {}\n'.format(left, right))

    for costs in model.unaries:
        f.write('\n{}\n'.format(len(costs)))
        f.write(' '.join('{:e}'.format(c) for c in costs))
        f.write('\n')

    for left, right, costs in model.pairwise:
        f.write('\n{}\n'.format(len(costs)))
        f.write(' '.join('{:e}'.format(c) for c in numpy.nditer(costs, order='C')))
        f.write('\n')
