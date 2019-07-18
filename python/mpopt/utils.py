import lzma


def smart_open(filename, *args, **kwargs):
    if filename.endswith('.xz'):
        return lzma.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)


def breadth_first_search(number_of_nodes, edges):
    result = []
    active = []
    inactive = set(range(number_of_nodes))

    while inactive:
        assert not active
        active.append(inactive.pop())

        while active:
            i = active.pop(0)
            result.append(i)

            neighs = set()
            for edge in edges:
                if i in edge:
                    for j in edge:
                        if j != i and j in inactive:
                            neighs.add(j)

            for j in neighs:
                active.append(j)
                inactive.remove(j)

    assert not active
    assert not inactive
    assert list(range(number_of_nodes)) == sorted(result)

    return result


def create_permutation(data):
    assert list(range(len(data))) == sorted(data)

    result = [None for _ in data]
    for i, x in enumerate(data):
        result[x] = i

    assert all(x is not None for x in result)
    return result
