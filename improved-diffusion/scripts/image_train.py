"""
image_train.py
Train a diffusion model on images.
"""

import argparse
import random
import torch as th
import numpy as np


from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def main():
    args = create_argparser().parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)

    try:
        dist_util.setup_dist()
    except Exception:
        logger.log("Distributed setup skipped. Proceeding with single-device training.")

    logger.configure()

    # Add the FP16 safeguard
    if args.use_fp16 and not th.cuda.is_available():
        logger.log("FP16 not enabled or supported, proceeding with FP32.")
        args.use_fp16 = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.to(dist_util.dev())
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model.to(device)
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()



def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()