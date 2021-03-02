import toml

from .CSA import CSA


def create_model(opt):
    header_kw
    model = CSA()
    model.initialize(opt)
    print(f"model [{model.name}] was created")
    return model
