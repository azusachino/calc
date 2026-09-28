"""Train the small character-level Transformer on data/input.txt."""

from pathlib import Path

import torch

from calc import config
from calc.model import LanguageModel
from calc.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "input.txt"
CHECKPOINT = ROOT / "artifacts" / "model.pt"


def main() -> None:
    text = DATA.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(tokens))
    train_data, validation_data = tokens[:split], tokens[split:]
    if len(train_data) <= config.BLOCK_SIZE or len(validation_data) <= config.BLOCK_SIZE:
        raise ValueError("corpus must contain at least 10 * BLOCK_SIZE characters")

    def batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
        x = torch.stack([data[i : i + config.BLOCK_SIZE] for i in starts])
        y = torch.stack([data[i + 1 : i + config.BLOCK_SIZE + 1] for i in starts])
        return x, y

    model = LanguageModel(
        tokenizer.vocab_size,
        config.BLOCK_SIZE,
        config.EMBEDDING_SIZE,
        config.NUM_HEADS,
        config.NUM_LAYERS,
        config.DROPOUT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.train()
    for step in range(config.STEPS):
        x, y = batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == config.STEPS - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(validation_data)
                _, validation_loss = model(vx, vy)
            print(f"step {step:4d}: train {loss.item():.3f}, val {validation_loss.item():.3f}")
            model.train()

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "tokens": tokenizer.tokens,
            "config": {
                "block_size": config.BLOCK_SIZE,
                "embedding_size": config.EMBEDDING_SIZE,
                "num_heads": config.NUM_HEADS,
                "num_layers": config.NUM_LAYERS,
                "dropout": config.DROPOUT,
            },
        },
        CHECKPOINT,
    )
    print(f"saved {CHECKPOINT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
