"""Command-line interface for FoodStack."""

import click


@click.group()
def main():
    """FoodStack: A voice-enabled multi-agent food delivery assistant."""
    pass


@main.command()
@click.option("--name", default="User", help="Name to greet")
def hello(name):
    """Say hello."""
    click.echo(f"Hello {name}! Welcome to FoodStack.")


if __name__ == "__main__":
    main()
