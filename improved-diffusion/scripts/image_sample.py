"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    # Set seed for reproducibility
    SEED = 42
    th.manual_seed(SEED)
    np.random.seed(SEED)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(SEED)

    # Initialize logger without distributed setup
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )

    # Set model to appropriate device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # Convert to HWC format
        sample = sample.contiguous()

        # No need for distributed gathering if only one GPU is available
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Combine images and labels into arrays
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    
    # Save generated samples
    shape_str = "x".join([str(x) for x in arr.shape])
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    out_path = os.path.join(desktop_path, f"openai_samples_cosine_learnedsigma_bcndataset{shape_str}.npz")

    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=600,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
