import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .transformer import Transformer
from .vllm import Vllm

MODEL_REGISTRY = {
    "transformers": Transformer,
    "vllm": Vllm,
}


def get_model(metric_name):
    return MODEL_REGISTRY[metric_name]
