"""``ds-train``: train the odor decoder, in three steps that each stand alone.

    1  pretraining   ``pretrain``     the decoder learns the language of molecules (no odor tokens)
    2  generator     ``generator``    odor tokens -> molecule; fine-tunes the pretrained decoder, loss on the molecule
    3  classifier     ``classifier``   molecule -> odor tokens; a head on the frozen pretrained decoder

Steps 2 and 3 both start from the pretrained decoder of step 1 and do not depend on each other: the
classifier never changes a weight of the decoder, and the generator does not train for naming. Each step
keeps its own folder (``stage1``, ``stage2``, ``stage3``), its own ``history.csv`` and its own scores.

    ds-train                    all three, one after the other
    ds-train --stage 1          one step
    ds-train --size small       a smaller model, in ``output/odor_small``

Every default of the generator is the best setting found, so a plain ``--stage 2`` trains the best model; its
options change one thing from it.
"""

from __future__ import annotations

import argparse

from scripts.core import classifier, data, generator, pretrain, training
from scripts.core import model as model_module
from scripts.core.history import logged
from scripts.core.vocabulary import Vocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the odor decoder.")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="stop each step after this many epochs"
    )
    parser.add_argument(
        "--size", choices=tuple(model_module.SIZES), default="base", help="the size of the model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=training.SEED,
        help="the random seed, to see how much a result moves between runs of the same settings",
    )
    parser.add_argument(
        "--attention-budget",
        type=int,
        default=None,
        help="the attention budget that caps a batch of long texts (default 64 x 160 x 160)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="the most texts in a batch (default 256)"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="keep this run apart, in stage2_<tag> and stage3_<tag>, to compare runs",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="train on the validation molecules too, for the last model once the settings are fixed "
        "(they are then no longer held out; the test and DREAM molecules still are)",
    )
    group = parser.add_argument_group("step 2, the generator")
    group.add_argument(
        "--full-rate-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train the token table, and so the odor tokens, at the full learning rate",
    )
    group.add_argument(
        "--replay",
        type=int,
        default=None,
        help="molecules of the pretraining mixed into every epoch, after an empty request "
        "(default: as many as there are molecules with odor tokens; 0: none)",
    )
    group.add_argument("--dropout", type=float, default=generator.DROPOUT, help="the dropout")
    group.add_argument(
        "--layer-decay",
        type=float,
        default=generator.LAYER_DECAY,
        help="learning rate of each layer as a share of the one above (1: all layers alike)",
    )
    group.add_argument("--lr", type=float, default=None, help="the peak learning rate")
    group.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="score and save the moving average of the weights, with this decay (0: off)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_module.use_size(args.size)
    if args.attention_budget:
        data.use_budget(args.attention_budget)
    if args.batch_size:
        data.use_batch_size(args.batch_size)
    if args.size != "base":
        training.OUTPUT_DIR = training.OUTPUT_DIR.with_name(
            f"{training.OUTPUT_DIR.name}_{args.size}"
        )
    training.SEED = (
        args.seed
    )  # the epochs' own random draws read it, so a different seed changes them too
    training.seed_everything(args.seed)
    vocabulary = Vocabulary.load()
    with logged(training.OUTPUT_DIR / "train.log"):
        print(f"device: {training.DEVICE}, {args.size} model")
        if args.stage in ("1", "all"):
            pretrain.train(vocabulary, args.max_epochs)
        if args.stage in ("2", "all"):
            generator.train(
                vocabulary,
                args.max_epochs,
                args.tag,
                args.full_rate_embeddings,
                args.dropout,
                None if args.layer_decay >= 1.0 else args.layer_decay,
                args.lr,
                args.replay,
                args.ema,
                args.include_validation,
            )
        if args.stage in ("3", "all"):
            classifier.train(
                vocabulary,
                args.max_epochs,
                tag=args.tag,
                include_validation=args.include_validation,
            )
    return 0
