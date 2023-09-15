from util.tools import load_model
import argparse
from os.path import basename

parser = argparse.ArgumentParser(description='Describe a model loaded from disk')
parser.add_argument('--model_path', type=str, required=True, help='Path to file containing the model')
args = parser.parse_args()

model = load_model(args.model_path)

print('\n{0}: \n--------------'.format(basename(args.model_path)))
print('Total training population size: {0}'.format(model.num_tokens))
print('Minimum training size: {0}'.format(model.min_training_samples))
print('Feature selection method: {0}'.format(model.feature_selection))
print('Input size: {0}'.format(model.feature_length))
print('Trained with loss function: {0}'.format(type(model.loss_object)))
print('Parameters: {:,}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('Bucket means: {0}\n'.format(model.bucket_means))

for trait in ['emb_size', 'num_heads', 'num_layers', 'hidden_dim', 'dropout', 'input_ln']:
    if hasattr(model, trait):
        print('{0}: {1}'.format(trait, getattr(model, trait)))
