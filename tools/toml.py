import toml


class Bunch(dict):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__dict__ = self


def load_option(path):
    opt = toml.load(path)
    return Bunch(opt)
