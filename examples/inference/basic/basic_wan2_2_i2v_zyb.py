from fastvideo import VideoGenerator
import torch
import time
import argparse
import logging

# from fastvideo.configs.sample import SamplingParam

# OUTPUT_PATH = "video_samples_wan2_2_14B_i2v"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to generate the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    
    args = parser.parse_args()
    return args
    
def main():
    args = _parse_args()
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        args.ckpt_dir,  # Wan-AI/Wan2.2-I2V-A14B-Diffusers
        # FastVideo will automatically handle distributed setup
        num_gpus=args.num_gpus,
        use_fsdp_inference=args.dit_fsdp,
        dit_cpu_offload=False,  # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )


    with torch.no_grad():
        _ = generator.generate_video(
            args.prompt,
            image_path=args.image,
            output_path=args.save_file,
            save_video=False,
            num_inference_steps=args.sample_steps,
            height=1280,
            width=720,
            seed=args.base_seed,
            num_frames=81,
        )

    start_time = time.time()
    # 记录开始时间
    video = generator.generate_video(
            args.prompt,
            image_path=args.image,
            output_path=args.save_file,
            save_video=True,
            num_inference_steps=args.sample_steps,
            height=1280,
            width=720,
            seed=args.base_seed,
            num_frames=81,
        )
    
    torch.cuda.synchronize()
    # 记录结束时间
    inference_time_ms = (time.time() - start_time)
    print(f"Inference time in FastVideo: {inference_time_ms:.2f} s")


if __name__ == "__main__":
    main()
