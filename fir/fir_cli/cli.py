import click
from .pipeline import pipeline


@click.group()
def cli():
    """Feature Importance Ranking CLI."""
    pass


cli.add_command(pipeline, name="run-pipeline")
