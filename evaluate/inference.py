from util.inference import forward
from util.tools import normalize, load_model
import torch
from joblib import load
import numpy as np
import pandas as pd
import random
import argparse
from os import path

# Manual seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Run a training and testing population and write results to csv')
parser.add_argument('--model_path', type=str, required=True, help='Path to file containing the model')
parser.add_argument('--target_path', type=str, required=True, help='Data to perform inference on (binary file).')
parser.add_argument('--train_path', type=str, required=True, help='Training data (binary file).')
parser.add_argument('--discrete_predictions', required=False, default=False, action='store_true')
parser.add_argument('--docker', required=False, default=False, action='store_true',
                    help='Flag for running in docker (changes data path)')
args, _ = parser.parse_known_args()

model_path = args.model_path
target_path = args.target_path
train_path = args.train_path

if args.docker:
    model_path = path.join('/deploy', path.basename(model_path))
    target_path = path.join('/data', path.basename(target_path))
    train_path = path.join('/data', path.basename(train_path))

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, device)
model.eval()

# Load the data
train_x, train_y, _ = load(train_path)
train_x = np.array(train_x).astype(np.float32)
train_y = normalize(np.array(train_y))

eval_x, _, eval_taxa = load(target_path)
eval_x = np.array(eval_x).astype(np.float32)

print('Training samples: {0}'.format(train_x.shape[0]))
print('Eval samples: {0}'.format(eval_x.shape[0]))

# Do the forward pass
pred_y = forward(train_x, train_y, eval_x, model, continuous=not args.discrete_predictions)

output_path = path.join(path.dirname(target_path), '{0}-predicted.csv'.format(path.basename(train_path)))
pd.DataFrame(list(zip(eval_taxa, pred_y))).to_csv(output_path, header=False, index=False)

print('Success! Taxa and their predictions for the target dataset were written to {0} \N{beer mug}'.format(output_path))
