import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval", "train"],
        default = "train",
        help="Which mode to run"
    )
    args = parser.parse_args()

    if args.mode == "train":
        from denoiser.training.trainer import Trainer

        trainer = Trainer()
        trainer.run()
    else:
        from denoiser.training.evaluator import run_evaluation

        run_evaluation()