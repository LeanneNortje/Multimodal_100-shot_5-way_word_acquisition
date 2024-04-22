def load(path, parser):
    with open(path, "r") as f:
        return list(map(parser, f.readlines()))