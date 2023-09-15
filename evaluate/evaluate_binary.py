from util.inference import forward
from util.tools import get_random_mask, normalize, load_model
from util.gblup import GBLUP
from util.feature_selection import feature_reduce
import torch
from joblib import load
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import random
import argparse
from os.path import basename


def inference_comparison(args, do_plot):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    model.eval()

    if args.train_path is None:
        print('No training data provided, will use a random {0} train-test split'.format(args.train_split))

        # Load the inference dataset
        data = load(args.target_path)

        if args.split_meta:
            print('Splitting on top-level list')
            available_families = list(range(len(data)))
            mask = get_random_mask(len(available_families), number_unmasked=int(len(available_families) * args.train_split))
            train_families = np.array(available_families)[np.array(mask).astype(bool)].tolist()

            train_x = np.concatenate([data[i][0] for i in available_families if i in train_families])
            train_y = np.concatenate([data[i][1] for i in available_families if i in train_families])
            eval_x = np.concatenate([data[i][0] for i in available_families if i not in train_families])
            eval_y = normalize(np.concatenate([data[i][1] for i in available_families if i not in train_families]))
        else:
            x = np.array(data[0])
            y = normalize(np.array(data[1]))
            del data

            print(x.shape)
            print(y.shape)

            training_mask = get_random_mask(x.shape[0], int(float(x.shape[0]) * args.train_split))

            train_x = x[training_mask == 1]
            eval_x = x[training_mask == 0]
            train_y = y[training_mask == 1]
            eval_y = normalize(y[training_mask == 0])
    else:
        train_data = load(args.train_path)
        train_x = np.array(train_data[0])
        train_y = normalize(np.array(train_data[1]))
        del train_data

        eval_data = load(args.target_path)
        eval_x = np.array(eval_data[0])
        eval_y = normalize(np.array(eval_data[1]))
        del eval_data

    print(basename(args.target_path))
    print('Training samples: {0}'.format(len(train_x)))
    print('Eval samples: {0}'.format(len(eval_x)))

    pred_y = forward(train_x, train_y, eval_x, model, continuous=not args.discrete_predictions)

    assert pred_y.shape == eval_y.shape

    pfn_p = pearsonr(eval_y, pred_y)[0]
    pfn_s = spearmanr(eval_y, pred_y).statistic

    print('PFN pearson r: {0:.4f}, spearman r: {1:.4f}'.format(pfn_p, pfn_s))

    # Do the evaluation for GBLUP
    gb = GBLUP()
    ebvs = gb.get_ebvs(train_x, eval_x, train_y)

    gblup_p = pearsonr(eval_y, ebvs)[0]
    gblup_s = spearmanr(eval_y, ebvs).statistic

    print('GBLUP pearson r: {0:.4f}, spearman r: {1:.4f}'.format(gblup_p, gblup_s))

    # Do the evaluation for PCR
    p_x, p_x_eval = feature_reduce(train_x, train_y, num_features=model.feature_length, method='pca', eval_x=eval_x)
    lr = LinearRegression()
    lr.fit(p_x, train_y)
    preds = lr.predict(p_x_eval)

    pcr_p = pearsonr(eval_y, preds)[0]
    pcr_s = spearmanr(eval_y, preds).statistic

    print('PCR r: {0:.4f}, spearman r: {1:.4f}'.format(pcr_p, pcr_s))

    if do_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        y_min = np.min([np.min(pred_y) - 1, np.min(ebvs) - 1])
        y_max = np.max([np.max(pred_y) + 1, np.max(ebvs) + 1])
        bins = np.arange(-4, 4, 0.1)

        axs[0][0].scatter(eval_y, pred_y, alpha=0.5)
        axs[0][0].set_ylim((y_min, y_max))
        axs[0][0].grid()
        axs[0][0].set_title('PFN')

        axs[0][1].scatter(eval_y, ebvs, alpha=0.5)
        axs[0][1].set_ylim((y_min, y_max))
        axs[0][1].grid()
        axs[0][1].set_title('GBLUP')

        axs[1][0].hist(pred_y, bins=bins, density=True)
        axs[1][0].grid()
        axs[1][0].set_title('Predicted eval y')

        axs[1][1].hist(eval_y.flatten(), bins=bins, density=True)
        axs[1][1].grid()
        axs[1][1].set_title('Actual eval y')

        plt.show()

    return pfn_p, gblup_p, pcr_p, pfn_s, gblup_s, pcr_s


if __name__ == '__main__':
    # Manual seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Run a population in binary format through a pre-trained model from disk')
    parser.add_argument('--model_path', type=str, required=True, help='Path to file containing the model')
    parser.add_argument('--target_path', type=str, required=True, help='Data to perform inference on')
    parser.add_argument('--train_path', type=str, required=False, default=None,
                        help='Training data. If not specified, a random split of the data will be used.')
    parser.add_argument('--train_split', type=float, required=False, default=0.8)
    parser.add_argument('--discrete_predictions', required=False, default=False, action='store_true')
    parser.add_argument('--split_meta', required=False, default=False, action='store_true')
    args = parser.parse_args()

    inference_comparison(args, do_plot=True)
