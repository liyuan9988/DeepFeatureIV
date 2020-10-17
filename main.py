import json
import logging
from typing import Optional
import click
from pathlib import Path
import uuid
from shutil import copyfile, make_archive
import os
from os import PathLike
import datetime


from src.utils.custom_logging import configure_logger
from src.experiment import experiments


DATA_DIR = Path.cwd().joinpath('data')
DUMP_DIR = Path.cwd().joinpath('dumps')
SRC_DIR = Path.cwd().joinpath('src')

SLACK_URL = None
NUM_GPU = None
if Path.cwd().joinpath('src/config.json').exists():
    SLACK_URL = json.load(Path.cwd().joinpath('src/config.json').open('r'))['slack']
    NUM_GPU = json.load(Path.cwd().joinpath('src/config.json').open('r')).get('num_gpu', None)

SCRIPT_NAME = Path(__file__).stem
LOG_DIR = Path.cwd().joinpath(f'logs/{SCRIPT_NAME}')

logger = logging.getLogger()


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--debug/--release', default=False)
@click.pass_context
def main(ctx, config_path, debug):
    if(debug):
        # Change logging level to debug
        logger.setLevel(logging.DEBUG)
        logger.handlers[-1].setLevel(logging.DEBUG)
        logger.debug("debug")

    foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    dump_dir = DUMP_DIR.joinpath(foldername)
    os.mkdir(dump_dir)
    with open(config_path) as f:
        config = json.load(f)
    ctx.obj["data_dir"] = dump_dir
    ctx.obj["config"] = config
    json.dump(config, open(dump_dir.joinpath("configs.json"), "w"))
    make_archive(dump_dir.joinpath("src"), "zip", root_dir=SRC_DIR)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def dfiv(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments("DFIV", config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def deepiv(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments("DeepIV", config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def kiv(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments("KernelIV", config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def rffkiv(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments("RFFKernelIV", config, data_dir, num_thread, NUM_GPU)

@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def deepgmm(ctx, num_thread):
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    experiments("deepGMM", config, data_dir, num_thread, NUM_GPU)


if __name__ == '__main__':
    configure_logger(SCRIPT_NAME, log_dir=LOG_DIR, webhook_url=SLACK_URL)
    try:
        main(obj={})
        logger.critical('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)