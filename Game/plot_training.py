import argparse
from pathlib import Path

from training_plots import generate_training_plots


def main() -> None:
        parser = argparse.ArgumentParser(
                description="Generate reward/loss training plots from metrics CSV"
        )
        parser.add_argument(
                "--metrics-file",
                type=str,
                required=True,
                help="Path to training metrics CSV file"
        )
        parser.add_argument(
                "--output-dir",
                type=str,
                default="training_figures",
                help="Directory to save generated figure (default: training_figures)"
        )
        args = parser.parse_args()

        output_file = generate_training_plots(
                metrics_file=args.metrics_file,
                output_dir=args.output_dir
        )

        if output_file is None:
                print("Failed to generate figure. Check metrics path and matplotlib installation.")
        else:
                print(f"Saved to {output_file}")

if __name__ == "__main__":
        main()