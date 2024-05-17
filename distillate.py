#!/usr/bin/env python
# coding: utf-8
import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch.cuda

from v_diffusion import make_beta_schedule
from train_utils import *
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import from_path
import wandb
import PIL
from log_utils import log_predictions, log_spectrogram


def on_iter(logs):
    if logs.get('log_audio'):
        log_predictions(wandb, logs['preds'], logs['target'], logs['audio_path'], logs)
    wandb.log({
        "train_loss": logs['loss'],
        "learning_rate": logs['current_lr']
    })


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.",
                        type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.",
                        type=str, required=True)
    parser.add_argument("--base_checkpoint", help="Path to base checkpoint.", type=str, required=True)
    parser.add_argument("--gamma", help="Gamma factor for SNR weights.", type=float, default=0)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=5000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.3 * 5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--model_params", type=str, default=None)
    parser.add_argument("--n_timesteps", type=int, default=0)
    parser.add_argument("--time_scale", type=float, default=0)
    parser.add_argument("--params_type", type=str, default='diffwave')
    return parser


def distill_model(args, make_model, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    need_student_ema = True
    if args.scheduler.endswith("SWA"):
        need_student_ema = False

    # print(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # train_dataset = test_dataset = InfinityDataset(make_dataset(), args.num_iters)

    # len(train_dataset), len(test_dataset)

    # img, anno = train_dataset[0]

    teacher_ema = make_model().to(device)

    # image_size = teacher_ema.image_size

    # checkpoints_dir = os.path.join("/kaggle/working/checkpoints", args.name, args.dname)
    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    ckpt = torch.load(args.base_checkpoint, map_location=device)
    # teacher_ema.load_state_dict(ckpt["G"])
    teacher_ema.load_state_dict(ckpt["model"])
    if args.n_timesteps == 0:
        n_timesteps = 1024
    else:
        n_timesteps = args.n_timesteps

    if args.time_scale == 0:
        time_scale = 1
    else:
        time_scale = args.time_scale

    # n_timesteps = ckpt["n_timesteps"]

    # time_scale = ckpt["time_scale"]
    # time_scale = args.time_scale
    del ckpt
    print(f"Num timesteps: {n_timesteps}, time scale: {time_scale}.")

    def make_scheduler():
        M = importlib.import_module("train_utils")
        D = getattr(M, args.scheduler)
        return D()

    scheduler = make_scheduler()
    distillation_model = DiffusionDistillation(scheduler, n_timesteps=n_timesteps)

    def make_diffusion(model, n_timesteps, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timesteps).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        r = D(model, betas, {"time_scale": time_scale, "n_timesteps": n_timesteps})
        r.time_scale = time_scale
        r.gamma = args.gamma
        return r

    teacher_ema_diffusion = make_diffusion(teacher_ema, n_timesteps, time_scale, device)

    student = make_model().to(device)
    if need_student_ema:
        student_ema = make_model().to(device)
    else:
        student_ema = None

    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        student.load_state_dict(ckpt["G"])
        student_ema.load_state_dict(ckpt["G"])
        del ckpt

    # distill_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.dataset != '':
        distill_train_loader = from_path(args.dataset, batch_size=args.batch_size, params_type=args.params_type)
    else:
        distill_train_loader = from_path('/kaggle/input/ljspeech-extended/LJSpeech/wavs', batch_size=args.batch_size)
    # tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    if args.checkpoint_to_continue == "":
        init_ema_model(teacher_ema, student, device)
        init_ema_model(teacher_ema, student_ema, device)
        print("Teacher parameters copied.")
    else:
        print("Continue training...")
    student_diffusion = make_diffusion(student, teacher_ema_diffusion.num_timesteps // 2,
                                       teacher_ema_diffusion.time_scale * 2, device)
    if need_student_ema:
        student_ema_diffusion = make_diffusion(student_ema, teacher_ema_diffusion.num_timesteps // 2,
                                               teacher_ema_diffusion.time_scale * 2, device)

        # Initialize Weights & Biases
    wandb.init(project="distillation_debug", config=args)

    # if args.checkpoint_to_continue == "":
    #     init_ema_model(teacher_ema, student, device)
    #     if student_ema:
    #         init_ema_model(teacher_ema, student_ema, device)
    #     print("Teacher parameters copied.")
    # else:
    #     print("Continue training...")

    # Define custom on_iter callback for wandb logging

    distillation_model.train_student(distill_train_loader, teacher_ema_diffusion,
                                     student_diffusion, student_ema, args.lr,
                                     device, make_extra_args=make_condition, on_iter=on_iter)

    print("Finished.")
    torch.save(student_diffusion, '/kaggle/working/out_model.pt')


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    distill_model(args, make_model, make_dataset)
