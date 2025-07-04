import hashlib
import urllib
import warnings
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from configs.constants import MODELS

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


def _download(url: str, root: str = str(Path.home() / ".cache" / "clip")):
    Path(root).mkdir(parents=True, exist_ok=True)
    filename = Path(url).name

    expected_sha256 = url.split("/")[-2]
    download_target = Path(root) / filename

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if download_target.is_file():
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(MODELS.keys())


# def load(
#     name: str,
#     device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
#     jit=False,
# ):
#     """Load a CLIP model

#     Parameters
#     ----------
#     name : str
#         A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

#     device : Union[str, torch.device]
#         The device to put the loaded model

#     jit : bool
#         Whether to load the optimized JIT model or more hackable non-JIT model (default).

#     Returns
#     -------
#     model : torch.nn.Module
#         The CLIP model

#     preprocess : Callable[[PIL.Image], torch.Tensor]
#         A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
#     """
#     if name in _MODELS:
#         model_path = _download(_MODELS[name])
#     elif Path(name).is_file():
#         model_path = name
#     else:
#         raise RuntimeError(
#             f"Model {name} not found; available models = {available_models()}"
#         )

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
#         state_dict = None
#     except RuntimeError:
#         # loading saved state dict
#         if jit:
#             warnings.warn(
#                 f"File {model_path} is not a JIT archive. Loading as a state dict instead"
#             )
#             jit = False
#         state_dict = torch.load(model_path, map_location="cpu")

#     if not jit:
#         model = build_model(state_dict or model.state_dict()).to(device)
#         if str(device) == "cpu":
#             model.float()
#         return model, _transform(model.visual.input_resolution)

#     # patch the device names
#     device_holder = torch.jit.trace(
#         lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
#     )
#     device_node = [
#         n
#         for n in device_holder.graph.findAllNodes("prim::Constant")
#         if "Device" in repr(n)
#     ][-1]

#     def patch_device(module):
#         try:
#             graphs = [module.graph] if hasattr(module, "graph") else []
#         except RuntimeError:
#             graphs = []

#         if hasattr(module, "forward1"):
#             graphs.append(module.forward1.graph)

#         for graph in graphs:
#             for node in graph.findAllNodes("prim::Constant"):
#                 if "value" in node.attributeNames() and str(node["value"]).startswith(
#                     "cuda"
#                 ):
#                     node.copyAttributes(device_node)

#     model.apply(patch_device)
#     patch_device(model.encode_image)
#     patch_device(model.encode_text)

#     # patch dtype to float32 on CPU
#     if str(device) == "cpu":
#         float_holder = torch.jit.trace(
#             lambda: torch.ones([]).float(), example_inputs=[]
#         )
#         float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
#         float_node = float_input.node()

#         def patch_float(module):
#             try:
#                 graphs = [module.graph] if hasattr(module, "graph") else []
#             except RuntimeError:
#                 graphs = []

#             if hasattr(module, "forward1"):
#                 graphs.append(module.forward1.graph)

#             for graph in graphs:
#                 for node in graph.findAllNodes("aten::to"):
#                     inputs = list(node.inputs())
#                     for i in [
#                         1,
#                         2,
#                     ]:  # dtype can be the second or third argument to aten::to()
#                         if inputs[i].node()["value"] == 5:
#                             inputs[i].node().copyAttributes(float_node)

#         model.apply(patch_float)
#         patch_float(model.encode_image)
#         patch_float(model.encode_text)

#         model.float()

#     return model, _transform(model.input_resolution.item())


def tokenize(
    tokenizer,
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    # import pdb
    # pdb.set_trace()
    if isinstance(texts, str):
        texts = [texts]  # ['a photo of a face.']

    sot_token = tokenizer.encoder["<|startoftext|>"]  # 49406
    eot_token = tokenizer.encoder["<|endoftext|>"]  # 49407
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  # 1,77

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:  # context_length 77
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
