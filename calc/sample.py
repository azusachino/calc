"""Generate a sample from artifacts/model.pt."""

from pathlib import Path

import torch

from calc import config
from calc.model import LanguageModel

CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts" / "model.pt"


def main() -> None:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    tokens = saved["tokens"]
    ids = {token: index for index, token in enumerate(tokens)}
    settings = saved["config"]
    model = LanguageModel(len(tokens), **settings)
    model.load_state_dict(saved["model"])
    model.eval()
    prompt = "\n"
    if prompt not in ids:
        prompt = tokens[0]
    start = torch.tensor([[ids[prompt]]])
    torch.manual_seed(config.SEED)
    sample = model.generate(start, max_new_tokens=500)[0].tolist()
    print("".join(tokens[index] for index in sample))


if __name__ == "__main__":
    main()
