import json
import re
from collections.abc import Iterable
from pathlib import Path


class SmilesTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    TOKEN_PATTERN = re.compile(
        r"\[[^\]]+\]|Br|Cl|se|as|%\d{2}|->|<-|[A-Z][a-z]?|[bcnops]|@@?|"
        r"\d|\.|[-=#$:*()\\/@+~?]"
    )

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {self.PAD: 0, self.UNK: 1}

    def tokenize(self, smiles: str) -> list[str]:
        tokens = self.TOKEN_PATTERN.findall(smiles)
        if "".join(tokens) != smiles:
            raise ValueError(f"Unrecognized SMILES syntax: {smiles!r}")
        return tokens

    def fit(self, smiles_strings: Iterable[str]) -> None:
        tokens = sorted({token for smiles in smiles_strings for token in self.tokenize(smiles)})
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.token_to_id.update({token: index + 2 for index, token in enumerate(tokens)})

    def encode(self, smiles: str, max_length: int) -> list[int]:
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")
        unknown_id = self.token_to_id[self.UNK]
        encoded = [self.token_to_id.get(token, unknown_id) for token in self.tokenize(smiles)]
        encoded = encoded[:max_length]
        padding = [self.token_to_id[self.PAD]] * (max_length - len(encoded))
        return encoded + padding

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        vocabulary = json.loads(path.read_text(encoding="utf-8"))
        if vocabulary.get(self.PAD) != 0 or vocabulary.get(self.UNK) != 1:
            raise ValueError("Tokenizer vocabulary has invalid PAD or UNK token IDs.")
        self.token_to_id = {str(token): int(index) for token, index in vocabulary.items()}
