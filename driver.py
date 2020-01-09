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


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-i",
    "--id",
    "id_",
    prompt="ID of the network",
    type=int,
    default=0,
    show_default=True,
)
def show(id_):
    """
    Show the structure of the network with the given ID
    """
    if id_ < 0 or id_ > 1000:
        print("invalid value")
    else:
        net = mnistk.networks.NET_LIST[id_]()
        print(net)


@main.command()
@click.option(
    "-i",
    "--id",
    "id_",
    prompt="ID of the network to run",
    type=int,
    default=0,
    show_default=True,
)
@click.option("-n", "--name", "name", type=str, default="t1", show_default=True)
@click.argument("params", type=click.File())
@click.argument(
    "dest",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
def run(id_, name, params, dest):
    """
    Obtain network ID, name of run,
    path of JSON for parameters, destination folder.

    :id: ID of the network to be used (0-1000)\n
    :name: name of the run for use in logging\n
    :params: file path of JSON containing parameters for run\n
    :dest: path of directory to store outputs\n
    """
    settings = mnistk.Settings(**json.load(params))
    trainer = mnistk.Trainer(settings, dest, net_id=id_, run_name=name)
    trainer.train()
    trainer.save()
    print(trainer._dest)


@main.command()
@click.argument(
    "result_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.argument(
    "csv_path", type=click.Path(resolve_path=True, file_okay=True, dir_okay=False)
)
def collect(result_dir, csv_path):
    """
    Collect results from JSON in subfolders into a single CSV

    :result_dir: Subdirectories of this directory contain the required JSON\n
    :csv_path: Path+filename of CSV to save\n
    """
    click.echo("Collecting results from JSONs to CSV ...")
    mnistk.write_to_csv(result_dir, csv_path)


if __name__ == "__main__":
    main()
