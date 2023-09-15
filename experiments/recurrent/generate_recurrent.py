"""Generate a full cached dataset without training"""

from loaders.simpop_prior import SimPopGenerator
from loaders.recurrent_prior import RecurrentGenerator
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Generate a full cached dataset without training')
parser.add_argument('--output_dir', type=str, required=True, help='The location to output cache files')
parser.add_argument('--num_threads', type=int, required=False, default=10, help='Number of simultaneous simulations.')
parser.add_argument('--feature_max_size', type=int, required=False, default=100,
                    help='Number of features for feature selection (ignored for \'g\')')
parser.add_argument('--num_samples', type=int, required=False, default=1000000, help='Total samples to output')
parser.add_argument('--pop_max_size', type=int, required=False, default=1000, help='Individuals per population')
parser.add_argument('--min_training_samples', type=int, required=False, default=None,
                    help='Minimum samples for training. If not supplied, will use half the total samples.')
parser.add_argument('--max_training_samples', type=int, required=False, default=None,
                    help='Maximum samples for training. If not supplied, will use the total samples minus ten.')
parser.add_argument('--feature_selection', type=str, required=False, default='pca',
                    choices=['greedy', 'correlation', 'mi', 'sequential', 'regression',
                             'rp', 'g', 'composite', 'pca', 'strided_pca', 'downsample'])
parser.add_argument('--mpi', required=False, default=False, action='store_true',
                    help='Set if running on multiple MPI nodes')
args, _ = parser.parse_known_args()

print('loaded')

min_training_samples = args.min_training_samples if args.min_training_samples is not None \
    else int(args.pop_max_size / 2)
max_training_samples = args.max_training_samples if args.max_training_samples is not None \
    else args.pop_max_size - 10

base_gen = SimPopGenerator(pop_max_size=args.pop_max_size, num_samples=args.num_samples,
                           feature_selection=None, verbose=False,
                           num_timesteps_min=50, num_timesteps_max=60,
                           use_cache=False)

breeding_prior = RecurrentGenerator(base_population_generator=base_gen,
                                    num_train_samples=max_training_samples,
                                    num_eval_samples=args.pop_max_size - min_training_samples,
                                    feature_selection=args.feature_selection,
                                    feature_max_size=args.feature_max_size,
                                    use_cache=True, cache_dir=args.output_dir)

if args.mpi:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor

    comm = MPI.COMM_WORLD


    def func(idx):
        _ = breeding_prior[idx]
        return True


    if __name__ == '__main__':
        print('Starting MPI pool (universe size: {0})'.format(comm.Get_attr(MPI.UNIVERSE_SIZE)))
        with MPIPoolExecutor() as executor:
            _ = executor.map(func, range(args.num_samples))
        print('MPI pool finished')

else:
    # Not running on MPI, we can just use a standard dataloader
    batch_size = 1
    loader = DataLoader(breeding_prior, batch_size=batch_size, num_workers=args.num_threads, shuffle=False)

    for i, _ in enumerate(loader):
        print('seen: {0} ({1:.2f}%)'.format(i * batch_size, 100. * (i * batch_size) / len(loader)))
