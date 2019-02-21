import lzma


def smart_open(filename, *args, **kwargs):
    if filename.endswith('.xz'):
        return lzma.open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)
