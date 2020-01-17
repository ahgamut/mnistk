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
def collect(result_dir):
    """
    Collect results from subfolders into a single CSV,
    and create rank-based data for each result

    :result_dir: Subdirectories of this directory contain the required JSON\n
    the generated file(s) will be stored at the top level of this directory.
    """
    mnistk.write_to_csv(result_dir)
    mnistk.save_exam_scores(result_dir)
    mnistk.save_rankings(result_dir)


if __name__ == "__main__":
    main()
