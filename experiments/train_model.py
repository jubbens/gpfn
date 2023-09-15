"""Train a meta-learner on various different priors"""

from loaders.simpop_prior import SimPopGenerator
from loaders.nam_prior import NAMGenerator
from loaders.silva_prior import SilvaGenerator
from loaders.recurrent_prior import RecurrentGenerator
from loaders.ril_prior import RILGenerator
from models.amortizedneuralgp import AmortizedNeuralGP
from util.bar_distribution import FullSupportBarDistribution, MSEFullSupportBarDistribution, get_bucket_limits
from util.optimization import get_cosine_schedule_with_warmup
from util.tools import save_model, eval_sampler
import torch
from torch.utils.data import DataLoader
from joblib import load, dump
import wandb
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import os.path
import random
import argparse
from datetime import datetime
import warnings

# Manual seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Train a meta-learner on various different priors', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prior', type=str, required=True, choices=['wild', 'nam', 'silva', 'recurrent', 'ril'], help='Which prior to fit (required)')
parser.add_argument('--device', type=str, required=False, default=None, help='Force a device to use')
parser.add_argument('--use_cache', required=False, default=False, action='store_true', help='Use a cached dataset, or output cached dataset after generating ')
parser.add_argument('--cache_dir', type=str, required=False, default='/tmp', help='The location to find or put cache files')
parser.add_argument('--num_threads', type=int, required=False, default=10, help='Number of simultaneous simulations.')
parser.add_argument('--feature_max_size', type=int, required=False, default=100, help='Number of features for feature selection')
parser.add_argument('--num_samples', type=int, required=False, default=400000, help='Total samples to output')
parser.add_argument('--pop_max_size', type=int, required=False, default=1000, help='Individuals per population')
parser.add_argument('--feature_selection', type=str, required=False, default='pca',
                    choices=['greedy', 'correlation', 'mi', 'sequential', 'regression',
                             'rp', 'g', 'composite', 'pca', 'strided_pca', 'downsample'], help='Strategy for feature selection')
parser.add_argument('--num_layers', type=int, required=False, default=12, help='Number of layers in transformer')
parser.add_argument('--hidden_dim', type=int, required=False, default=2048, help='Size of fc layers in transformer')
parser.add_argument('--num_heads', type=int, required=False, default=1, help='Number of heads for multi-head attention')
parser.add_argument('--no_input_norm', required=False, default=False, action='store_true', help='Forego input normalization')
parser.add_argument('--emb_size', type=int, required=False, default=2048, help='Dimensionality of input embedding')
parser.add_argument('--dropout_p', type=float, required=False, default=0., help='Dropout rate for the transformer.')
parser.add_argument('--min_training_samples', type=int, required=False, default=None,
                    help='Minimum samples for training. If not supplied, will use half the total samples.')
parser.add_argument('--max_training_samples', type=int, required=False, default=None,
                    help='Maximum samples for training. If not supplied, will use the total samples minus ten.')
parser.add_argument('--nonuniform_eval_pos', required=False, default=False, action='store_true',
                    help='Bias sampling towards larger train sets.')
parser.add_argument('--batch_size', type=int, required=False, default=8, help='Samples per batch')
parser.add_argument('--lr', type=float, required=False, default=0.00005, help='Learning rate')
parser.add_argument('--warmup_steps', type=int, required=False, default=25000, help='Warmup steps for optimizer')
parser.add_argument('--accumulation_steps', type=int, required=False, default=16, help='Number of steps to accumulate gradients')
parser.add_argument('--loss_function', type=str, required=False, default='bar_mse', choices=['bar', 'bar_mse'], help='Loss function to use')
parser.add_argument('--allow_tf32', required=False, default=False, action='store_true',
                    help='Allow Tensor-Float32 precision for cuda matmul ops')
parser.add_argument('--num_buckets', type=int, required=False, default=50,
                    help='Number of buckets for discretizing output')
parser.add_argument('--save_path', type=str, required=False, default=None,
                    help='Location to save the checkpoints and final model.')
parser.add_argument('--do_log', required=False, default=False, action='store_true', help='Output to wandb')
parser.add_argument('--log_steps', type=int, required=False, default=100, help='Log every n steps')
parser.add_argument('--checkpoint_steps', type=int, required=False, default=None, help='Save a checkpoint every n steps')
parser.add_argument('--asynchronous', required=False, default=False, action='store_true',
                    help='If set, the dataloader watches the cache directory and waits for data to appear.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)

if device.type == 'cuda' and args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

min_training_samples = args.min_training_samples if args.min_training_samples is not None \
    else int(args.pop_max_size / 2)
max_training_samples = args.max_training_samples if args.max_training_samples is not None \
    else args.pop_max_size - 10

if args.feature_selection == 'composite':
    expected_input_size = args.pop_max_size + args.feature_max_size
else:
    expected_input_size = args.feature_max_size


if args.prior == 'wild':
    prior = SimPopGenerator(pop_max_size=args.pop_max_size, num_samples=args.num_samples,
                            feature_max_size=args.feature_max_size, feature_selection=args.feature_selection,
                            verbose=False, use_cache=args.use_cache, cache_dir=args.cache_dir,
                            asynchronous=args.asynchronous)
elif args.prior == 'nam':
    base_gen = SimPopGenerator(pop_max_size=[200, 500], num_samples=args.num_samples,
                               feature_selection=None, verbose=False)

    prior = NAMGenerator(base_population_generator=base_gen,
                         num_families_min=25, num_families_max=85,
                         num_train_samples=max_training_samples, num_eval_samples=args.pop_max_size - min_training_samples,
                         feature_max_size=args.feature_max_size, feature_selection=args.feature_selection,
                         use_cache=args.use_cache, cache_dir=args.cache_dir, asynchronous=args.asynchronous)
elif args.prior == 'ril':
    base_gen = SimPopGenerator(pop_max_size=[200, 500], num_samples=args.num_samples,
                               feature_selection=None, verbose=False)

    prior = RILGenerator(base_population_generator=base_gen,
                         pop_max_size=args.pop_max_size,
                         feature_max_size=args.feature_max_size, feature_selection=args.feature_selection,
                         use_cache=args.use_cache, cache_dir=args.cache_dir, asynchronous=args.asynchronous)
elif args.prior == 'silva':
    base_gen = SimPopGenerator(num_samples=args.num_samples, feature_selection=None, verbose=False)

    prior = SilvaGenerator(base_population_generator=base_gen,
                           num_families_min=50, num_families_max=60,
                           family_size_min=20, family_size_max=25,
                           num_train_samples=max_training_samples, num_eval_samples=args.pop_max_size - min_training_samples,
                           feature_max_size=args.feature_max_size, feature_selection=args.feature_selection,
                           use_cache=args.use_cache, cache_dir=args.cache_dir, asynchronous=args.asynchronous)
elif args.prior == 'recurrent':
    base_gen = SimPopGenerator(num_samples=args.num_samples, feature_selection=None, verbose=False)

    prior = RecurrentGenerator(base_population_generator=base_gen,
                               num_train_samples=max_training_samples, num_eval_samples=args.pop_max_size - min_training_samples,
                               pregenerations_min=3, pregenerations_max=8,
                               feature_max_size=args.feature_max_size, feature_selection=args.feature_selection,
                               use_cache=args.use_cache, cache_dir=args.cache_dir, asynchronous=args.asynchronous)
else:
    raise NotImplementedError('Unknown prior: {0}'.format(args.prior))

num_batches = -(args.num_samples // -args.batch_size)

# Make a schedule of eval poses to batches
# We have to do this before the fact so that the data generator threads
# and training loop are on the same page about where the eval pos should be
if args.use_cache and os.path.exists(os.path.join(args.cache_dir, 'metadata.bin')):
    # We're using a cached dataset which provides info about eval pos, use it
    print('Training using eval schedule metadata found in the cache directory')
    eval_sched = load(os.path.join(args.cache_dir, 'metadata.bin'))
else:
    print('Did not find metadata in the cache directory, will make a new eval schedule')
    eval_sched = np.repeat(np.array([eval_sampler(min_training_samples, max_training_samples, args.pop_max_size,
                                                  uniform=not args.nonuniform_eval_pos)
                                     for _ in range(num_batches)]), args.batch_size)

    if args.use_cache:
        # If we're saving the dataset, we should provide this information too
        dump(eval_sched, os.path.join(args.cache_dir, 'metadata.bin'))

prior.set_eval_sched(eval_sched)
loader = DataLoader(prior, batch_size=args.batch_size, num_workers=args.num_threads, shuffle=False)

print('Approximating y distribution...')
parfunc = lambda i: prior[i][1]
samples = np.concatenate(Parallel(n_jobs=cpu_count())(delayed(parfunc)(i) for i in range(10)))

if args.use_cache and args.asynchronous:
    # We can start deleting cache files if we're in asynchronous mode now that that's done
    prior.set_delete_cached()

if args.loss_function == 'bar':
    criteria = FullSupportBarDistribution(
        borders=get_bucket_limits(args.num_buckets, ys=torch.tensor(samples)).to(device))
elif args.loss_function == 'bar_mse':
    criteria = MSEFullSupportBarDistribution(
        borders=get_bucket_limits(args.num_buckets, ys=torch.tensor(samples)).to(device))
else:
    raise NotImplementedError('Unknown loss function: {0}'.format(args.loss_function))

bucket_means = (criteria.borders[:-1] + criteria.bucket_widths / 2).cpu().numpy()

model = AmortizedNeuralGP(expected_input_size, n_out=args.num_buckets,
                          emb_size=args.emb_size, num_layers=args.num_layers,
                          hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                          dropout=args.dropout_p, input_ln=not args.no_input_norm)

# Stick all of this info into the model for reference
model.set_bucket_means(bucket_means)
model.set_loss_object(criteria)
model.set_num_tokens(args.pop_max_size)
model.set_feature_selection(args.feature_selection)
model.set_min_training_samples(min_training_samples)

if device.type == 'cuda':
    model = torch.nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_steps / args.batch_size), num_batches)

# wandb stuff
if args.do_log:
    wandb.init(
        project='gpfn',
        config={
            'prior': args.prior,
            'feature_max_size': args.feature_max_size,
            'feature_selection': args.feature_selection,
            'num_layers': args.num_layers,
            'hidden_dim': args.hidden_dim,
            'num_heads': args.num_heads,
            'emb_size': args.emb_size,
            'dropout_p': args.dropout_p,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'accumulation_steps': args.accumulation_steps,
            'loss_function': args.loss_function,
            'num_buckets': args.num_buckets,
            'nonuniform_eval_pos': args.nonuniform_eval_pos
        }
    )

running_loss = []
running_r = []

for i, (x, y) in enumerate(loader):
    # Truncate the data if it's not the right feature size. This allows us
    # to use cached feature sets at a lower resolution.
    if x.shape[2] > args.feature_max_size:
        warnings.warn('Truncating datapoints from {0} to {1} features'.format(x.shape[2], args.feature_max_size))
        x = x[:, :, :args.feature_max_size]

    eval_starts = eval_sched[i * args.batch_size:(i * args.batch_size) + args.batch_size]
    assert len(np.unique(eval_starts)) == 1
    eval_start = eval_starts[0]

    #  Truncate eval samples to fit into max pop size
    x, y = x[:, :args.pop_max_size], y[:, :args.pop_max_size]

    x = x.transpose(1, 0)
    y = y.transpose(1, 0)

    out = model(x.to(device), y.to(device), eval_start)
    loss = torch.mean(criteria(out[eval_start:], y[eval_start:].to(device)))
    loss = loss / args.accumulation_steps

    # pearson r (just for display)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    pred = [target.loss_object.mean(out[eval_start:, j, :]).detach().cpu().numpy().flatten() for j in range(args.batch_size)]
    gt = [y[eval_start:, j].numpy().flatten() for j in range(args.batch_size)]
    r = np.mean([pearsonr(p, g)[0] for p, g in zip(pred, gt)])

    print('loss: {0:.2f} r: {1:.2f} seen: {2} ({3:.2f}%) lr: {4:.7f}'.format(loss.detach().cpu().numpy(),
                                                                             r,
                                                                             i * args.batch_size,
                                                                             (i * args.batch_size) / args.num_samples * 100.,
                                                                             scheduler.get_last_lr()[0]))

    loss.backward()

    # Gradient accumulation
    if ((i + 1) % args.accumulation_steps == 0) or (i in [len(loader) - 1, 0]):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    # wandb stuff
    if args.do_log:
        running_loss.append(loss.detach().cpu().numpy())
        running_r.append(r)

        if i % args.log_steps == 0 or i == len(loader) - 1:
            wandb.log({'loss': np.mean(running_loss), 'r': np.mean(running_r)}, step=i)
            running_loss = []
            running_r = []

    # Checkpointing
    if args.checkpoint_steps is not None and args.save_path is not None:
        if i > 0 and i % args.checkpoint_steps == 0:
            save_model(model, os.path.join(args.save_path, 'checkpoint-{0}'.format(i)))

if args.save_path is not None:
    now = datetime.now()
    save_model(model, os.path.join(args.save_path, now.strftime('final-%d.%m.%Y.%H.%M.pt')))
