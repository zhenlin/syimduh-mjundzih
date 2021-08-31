import numpy as np
import pandas as pd

import torch
from torch import optim
import torch.optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.utils.data

import tqdm

import argparse
import os
import itertools
import sys
import random

import sdmd.config
import sdmd.model
import sdmd.util

from typing import Any, Iterable, Sequence, Tuple


def train_one_epoch(encoder: nn.Module, decoder: nn.Module, vae_encoder: nn.Module, vae_decoder: nn.Module, criterion: nn.Module, features_criterion: nn.Module, optimizer: torch.optim.Optimizer, data_loader: torch.utils.data.DataLoader, device: torch.device, epoch: int, num_data: int = None) -> float:
    torch_dtype = torch.get_default_dtype()

    encoder.train()
    decoder.train()

    with tqdm.tqdm(desc=f'Epoch {epoch}') as pbar:
        if num_data is not None:
            pbar.total = num_data

        pbar.refresh()

        cum_loss = 0.0
        cum_weight = 0.0

        for batch_idx, (inputs, labels, targets) in enumerate(data_loader):
            batch_size = inputs.shape[0]

            inputs = inputs.to(device, dtype=torch_dtype)
            targets = targets.to(device, dtype=torch_dtype)

            features = encoder(inputs)

            latent_mu, latent_log_sigma = vae_encoder(features)

            latent, prior_log_like, post_log_like = sdmd.model.vae_get_sample(latent_mu, latent_log_sigma)

            output_features = vae_decoder(latent)

            outputs = decoder(output_features)

            target_features = encoder(targets)

            target_latent_mu, target_latent_log_sigma = vae_encoder(target_features)

            # loss = criterion(outputs, targets, prior_log_like, post_log_like)

            # loss = criterion(outputs, targets) + features_criterion(features, target_features)

            loss = sum((
                criterion(outputs, targets, prior_log_like, post_log_like),
                features_criterion(target_features, output_features),
                sdmd.model.normal_kl_divergence(target_latent_mu, target_latent_log_sigma, latent_mu, latent_log_sigma).flatten(1).sum(1).mean(),
            ))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_scalar = loss.item()

            cum_loss += batch_size * loss_scalar
            cum_weight += batch_size

            mean_loss = cum_loss / cum_weight

            pbar.set_postfix(mean_loss=f'{mean_loss:.2e}', lr=optimizer.param_groups[0]['lr'])
            pbar.update(batch_size)

    return mean_loss


def evaluate(encoder: nn.Module, decoder: nn.Module, vae_encoder: nn.Module, vae_decoder: nn.Module, criterion: nn.Module, features_criterion: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, num_data: int = None) -> None:
    torch_dtype = torch.get_default_dtype()

    encoder.eval()
    decoder.eval()

    with tqdm.tqdm(desc=f'Evaluating') as pbar, torch.no_grad():
        if num_data is not None:
            pbar.total = num_data

        pbar.refresh()

        losses = []

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            batch_size = inputs.shape[0]

            inputs = inputs.to(device, dtype=torch_dtype)

            features = encoder(inputs)

            latent_mu, latent_log_sigma = vae_encoder(features)

            latent, prior_log_like, post_log_like = sdmd.model.vae_get_mode(latent_mu, latent_log_sigma)

            output_features = vae_decoder(latent)

            outputs = decoder(output_features)

            roundtrip_features = encoder(outputs)

            roundtrip_latent_mu, roundtrip_latent_log_sigma = vae_encoder(roundtrip_features)

            # loss = criterion(outputs, inputs) + features_criterion(features, roundtrip_features)

            loss = sum((
                criterion(outputs, inputs, prior_log_like, post_log_like),
                features_criterion(features, output_features),
                sdmd.model.normal_kl_divergence(roundtrip_latent_mu, roundtrip_latent_log_sigma, latent_mu, latent_log_sigma).flatten(1).sum(1),
            ))

            losses.append(loss.cpu())

            pbar.update(batch_size)

    losses = torch.cat(losses).numpy()

    mean_loss = np.mean(losses)
    loss_stdev = np.std(losses)

    print(f'Mean loss: \t{mean_loss:.2e}')
    print(f'Std dev: \t{loss_stdev:.2e}')
    print(f' 1%ile loss: \t{np.quantile(losses, 0.01):.2e}')
    print(f'10%ile loss: \t{np.quantile(losses, 0.1):.2e}')
    print(f'25%ile loss: \t{np.quantile(losses, 0.25):.2e}')
    print(f'50%ile loss: \t{np.median(losses):.2e}')
    print(f'75%ile loss: \t{np.quantile(losses, 0.75):.2e}')
    print(f'90%ile loss: \t{np.quantile(losses, 0.9):.2e}')
    print(f'99%ile loss: \t{np.quantile(losses, 0.99):.2e}')

    return



def preload_glyphs(dataset: sdmd.model.SDMDDataset, indices: Iterable[int]) -> None:
    for k in tqdm.tqdm(indices, desc='Loading glyphs'):
        dataset[k]
    
    return


def main(args: Any) -> None:
    device = torch.device(args.device)

    size = args.size

    random.seed(141421356)
    torch.manual_seed(141421356)
    np.random.seed(141421356)

    generator = torch.default_generator

    dataset, train_idxs, test_idxs = sdmd.config.load_data(args.dataset, size, generator)

    if args.eval_recall:
        test_idxs = train_idxs

    if args.eval_fraction < 1:
        test_idxs = tuple(np.compress(np.diff(np.floor(np.arange(len(test_idxs)) * args.eval_fraction), prepend=0) == 1, test_idxs))

    num_workers = args.num_workers
    batch_size = args.batch_size

    train_dataset = sdmd.model.AugmentingDataset(torch.utils.data.Subset(dataset, train_idxs), size, random_translation_max=args.random_translation_max, random_hole_probability=args.random_hole_probability, random_hole_size_max=args.random_hole_size_max, return_original=True)
    test_dataset = sdmd.model.AugmentingDataset(torch.utils.data.Subset(dataset, test_idxs), size)

    train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True)
    train_num_data = len(train_dataset)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True)
    test_num_data = len(test_dataset)

    model_name = args.model

    if model_name == 'alexnet':
        encoder = sdmd.model.SDMDAlexNetEncoder()
        decoder = sdmd.model.SDMDAlexNetDecoder(size)
    else:
        raise ValueError(f'unknown model: {model_name}')

    vae_encoder = sdmd.model.VAEEncoder(256 * 6 * 6, 256, 512)
    vae_decoder = sdmd.model.VAEDecoder((256, 6, 6), 256, 512)

    num_epochs = args.num_epochs
    start_epoch = 1

    # criterion = sdmd.model.MultiScale2dBCELoss((1, 5, 17, 128, 192, 384))
    criterion = sdmd.model.MultiScale2dELBoLoss((1, 5, 17, 128, 192, 384))

    features_criterion = sdmd.model.MSELoss()

    # eval_criterion = sdmd.model.MultiScale2dBCELoss((1, 5, 17, 128, 192, 384), reduction='none')
    eval_criterion = sdmd.model.MultiScale2dELBoLoss((1, 5, 17, 128, 192, 384), reduction='none')

    eval_features_criterion = sdmd.model.MSELoss(reduction='none')

    encoder.to(device)
    decoder.to(device)

    vae_encoder.to(device)
    vae_decoder.to(device)

    params = itertools.chain(
        encoder.parameters(),
        decoder.parameters(),
        vae_encoder.parameters(),
        vae_decoder.parameters(),
        )

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience, threshold=1e-4, threshold_mode='rel')

    num_checkpoints = args.num_checkpoints

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_device = torch.device('cpu')

    checkpoint_path = os.path.join(checkpoint_dir, 'autoencoder-checkpoint.pt')
    checkpoint = None

    if args.no_resume:
        if args.resume_from is not None:
            raise ValueError('cannot specify both --no-resume and --resume-from')
    else:
        if args.resume_from is None:
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device)
        else:
            checkpoint = torch.load(args.resume_from, map_location=checkpoint_device)

    est_losses = []
    min_est_loss = np.inf
    must_rollback = False

    if checkpoint:
        resume_strict = not args.resume_nonstrict

        encoder.load_state_dict(checkpoint['encoder'], strict=resume_strict)
        decoder.load_state_dict(checkpoint['decoder'], strict=resume_strict)

        vae_encoder.load_state_dict(checkpoint['vae_encoder'], strict=resume_strict)
        vae_decoder.load_state_dict(checkpoint['vae_decoder'], strict=resume_strict)

        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            est_losses = list(checkpoint['est_losses'])
            min_est_loss = min(est_losses)

    if args.eval_only:
        epochs = ()
    else:
        epochs = range(start_epoch, num_epochs + 1)

    if args.preload_glyphs and len(epochs) > 0:
        preload_glyphs(dataset, train_idxs)

    for epoch in epochs:
        est_loss = train_one_epoch(encoder, decoder, vae_encoder, vae_decoder, criterion, features_criterion, optimizer, train_data_loader, device, epoch, train_num_data)

        if est_loss < min_est_loss:
            min_est_loss = est_loss
        elif not np.isfinite(est_loss):
            must_rollback = True
        elif len(est_losses) >= 20:
            est_loss_rank = sum(1 if est_loss > x else 0 for x in est_losses)

            if 4 * est_loss_rank >= len(est_losses):
                must_rollback = True
        
        if must_rollback:
            print('Emergency stop. Please reload from last checkpoint.', file=sys.stderr)
            return

        est_losses.append(est_loss)
        lr_scheduler.step(est_loss)

        checkpoint = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'vae_encoder': vae_encoder.state_dict(),
            'vae_decoder': vae_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'est_losses': est_losses,
        }

        new_checkpoint_name = f'autoencoder-checkpoint-{epoch % num_checkpoints}.pt'

        torch.save(checkpoint, os.path.join(checkpoint_dir, new_checkpoint_name))

        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
        os.symlink(new_checkpoint_name, checkpoint_path)

    if args.preload_glyphs:
        preload_glyphs(dataset, test_idxs)

    evaluate(encoder, decoder, vae_encoder, vae_decoder, eval_criterion, eval_features_criterion, test_data_loader, device, test_num_data)

    return


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Trains a model.')

    parser.add_argument('-i', '--dataset', default='data/rs-unicode-primary.json', type=str, help='path to dataset configuration')
    parser.add_argument('-m', '--model', default='alexnet', help='model to train')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), type=str, help='device to use')
    parser.add_argument('-j', '--num-workers', default=min(3, os.cpu_count() - 1), type=int, help='number of DataLoader workers')

    parser.add_argument('-s', '--size', default=384, type=int, help='canvas height/width (in pixels)')
    parser.add_argument('--random-translation-max', default=8, type=int, help='maximum horizontal/vertical displacement for random translation')
    parser.add_argument('--random-hole-probability', default=0.5, type=float, help='probability of random deletion')
    parser.add_argument('--random-hole-size-max', default=96, type=int, help='maximum height/width of randomly deleted block')
    parser.add_argument('--preload-glyphs', action='store_true', help='load all required glyphs in advance')

    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='total number of epochs to run')
    parser.add_argument('-l', '--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('-p', '--momentum', default=0.9, type=float, help='momentum factor for stochastic gradient descent')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='weight decay for stochastic gradient descent')
    parser.add_argument('--lr-decay-factor', default=0.1, type=float, help='factor by which to decrease learning rate')
    parser.add_argument('--lr-patience', default=8, type=float, help='number of epochs to wait for improvement')

    parser.add_argument('-o', '--checkpoint-dir', default='checkpoints', type=str, help='directory in which to write checkpoints')
    parser.add_argument('-k', '--num-checkpoints', default=10, type=int, help='number of checkpoints to keep')

    parser.add_argument('--no-resume', action='store_true', help='do not attempt to resume from last checkpoint')
    parser.add_argument('-r', '--resume-from', default=None, type=str, help='path to checkpoint to resume from')
    parser.add_argument('-R', '--resume-model-only', action='store_true', help='load only model parameters from checkpoint')
    parser.add_argument('--resume-nonstrict', action='store_true', help='ignore missing or unexpected parameters in checkpoint')

    parser.add_argument('-t', '--eval-only', action='store_true', help='only evaluate the model')
    parser.add_argument('--eval-recall', action='store_true', help='use the training set for evaluation (instead of the test set)')
    parser.add_argument('--eval-fraction', default=1.0, type=float, help='fraction of evaluation set to use')

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)