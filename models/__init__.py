from . import base
from . import denseformer
from . import connect_to_last
from . import base_w_gains

MODELS = {
    "denseformer": denseformer.DenseFormer,
    "base": base.GPTBase,
    "connecttolast": connect_to_last.GPTBase,
    "basewgains": base_w_gains.GPTBase,
}


def make_model_from_args(args):
    return MODELS[args.model](args)


def registered_models():
    return MODELS.keys()
