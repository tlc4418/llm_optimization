import typer
from src.ppo.custom_helpers import gold_score

if __name__ == "__main__":
    typer.run(gold_score)
