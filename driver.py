"""
File: driver.py
Author: Gautham Venkatasubramanian
Email: ahgamut@gmail.com
Github: github.com/ahgamut
Description: Driver for the mnistk package
"""
import click
import json
import mnistk


@click.command()
@click.option("-i", "--id", "id_", type=int, default=0, show_default=True)
@click.option("-r", "--run", "run", type=str, default="t1", show_default=True)
@click.argument("params", type=click.File())
@click.argument(
    "dest",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
def main(id_, run, params, dest):
    """
    Obtain network ID, name of run,
    path of JSON for parameters, destination folder.

    :id: ID of the network to be used (0-1000)\n
    :run: name of the run for use in logging\n
    :params: file path of JSON containing parameters for run\n
    :dest: path of directory to store outputs\n
    """
    settings = mnistk.Settings(**json.load(params))
    trainer = mnistk.Trainer(settings, dest, net_id=id_, run_name=run)
    trainer.summary()
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
