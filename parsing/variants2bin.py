"""Convert a phenotype csv file and genotype hmp/vcf file to a .bin file appropriate for inference"""

import pandas as pd
import numpy as np
from joblib import dump
import argparse
from os import path
from math import isnan

parser = argparse.ArgumentParser(description='Generate a full cached dataset without training')
parser.add_argument('--genotype_file', type=str, required=True, help='Genotype file in either hapmap or VCF.')
parser.add_argument('--phenotype_file', type=str, required=False, default=None, help='Phenotype file in CSV format. Rows are taxa and columns are phenotypes.')
parser.add_argument('--phenotype_name', type=str, required=False, default=None, help='Name of the phenotype to use, should correspond to the column header.')
parser.add_argument('--ignore_missing', required=False, default=False, action='store_true', help='Force missing markers to homo major (not recommended).')
parser.add_argument('--docker', required=False, default=False, action='store_true', help='Flag for running in docker (changes data path)')
args, _ = parser.parse_known_args()

genotype_path = args.genotype_file
phenotype_path = args.phenotype_file

if phenotype_path is None:
    print('No phenotype file provided, will not include any phenotype information.')

if args.docker:
    genotype_path = path.join('/data', path.basename(genotype_path))

    if phenotype_path is not None:
        phenotype_path = path.join('/data', path.basename(phenotype_path))

print('Genotype path is {0}'.format(genotype_path))
if phenotype_path is not None:
    print('Phenotype path is {0}'.format(phenotype_path))

if '.vcf' in path.basename(genotype_path):
    print('It looks like the genotype file is VCF.')
    input_format = 'vcf'
elif '.hmp' in path.basename(genotype_path):
    print('It looks like the genotype file is hapmap.')
    input_format = 'hmp'
else:
    raise NotImplementedError('Not sure about the format of the genotype data, because it '
                              'does not seem to be .vcf or .hmp.txt.')

print('I\'m going to try reading the genotype file...', end='', flush=True)
geno_raw = pd.read_csv(genotype_path, sep='\s+')
print('\N{heavy check mark}')

if phenotype_path is not None:
    print('I\'m going to try reading the phenotype file...', end='', flush=True)
    pheno_raw = pd.read_csv(phenotype_path, sep='\s+')
    print('\N{heavy check mark}')

print('Okay, let\'s parse the genotype file.')

print('Trying to detect where the taxa start in the genotype file.')
if phenotype_path is not None:
    print('Grabbing the name of the first taxa from the phenotypes file, I assume it\'s in the first column.')
    first_taxa = pheno_raw.iloc[0, 0]
    first_col = list(geno_raw.columns).index(first_taxa)
else:
    max_col = 20

    for i in range(0, max_col):
        first_col = i
        d = geno_raw.iloc[0, i]
        if d in ['0/0', '1/1', '0/1', './.', './0', './1', '0/.', '1/.',
                 'A', 'B', 'C', 'T', 'G',
                 'R', 'M', 'S', 'W', 'Y', 'K', 'N',
                 'AB', 'BA',
                 'AA', 'TT', 'AT', 'TA',
                 'CC', 'GG', 'CG', 'GC'] and 'allele' not in list(geno_raw.columns)[i]:
            break

        if i == max_col:
            print('Tried to find the first data column, but probably failed. Going to use {0} as the first column but we '
                  'are probably missing some data.'.format(i))

print('Detected first col: {0}'.format(first_col))
print('That means the first taxa is {0}'.format(list(geno_raw.columns)[first_col]))

# Let's inspect the geno file and see if we can figure out how to parse it
if input_format == 'vcf':
    if args.ignore_missing:
        snp_dict = [{'0/0': 2, '1/1': 0, '0/1': 1,
                    './.': 2, './0': 2, './1': 2, '0/.': 2, '1/.': 2}] * geno_raw.index.size
    else:
        snp_dict = [{'0/0': 2, '1/1': 0, '0/1': 1,
                    './.': np.nan, './0': np.nan, './1': np.nan, '0/.': np.nan, '1/.': np.nan}] * geno_raw.index.size
elif input_format == 'hmp':
    # Read the alleles column to get reference/alternate for each site
    if 'alleles' not in list(geno_raw.columns):
        print('There is no alleles column in the hapmap file? \N{cross mark}')
        print(list(geno_raw.columns))
        exit()

    print('Reading the alleles column to figure out the encoding. I\'m going to assume it\'s something like A/B...',
          end='', flush=True)
    snp_desc = [p.split('/') for p in geno_raw['alleles']]
    print('\N{heavy check mark}')

    # Are the sites encoded with one or two characters?
    print('I\'m going to try and see if the sites are encoded with one or two characters.')
    ftg = geno_raw.iloc[0, first_col]

    if len(ftg) == 1:
        print('Looks like sites are encoded with a single character.')

        if args.ignore_missing:
            snp_dict = [{p[0]: 2, p[1]: 0, 'R': 1, 'M': 1, 'S': 1, 'W': 1, 'Y': 1, 'K': 1, 'N': 2} for p in snp_desc]
        else:
            snp_dict = [{p[0]: 2, p[1]: 0, 'R': 1, 'M': 1, 'S': 1, 'W': 1, 'Y': 1, 'K': 1, 'N': np.nan} for p in snp_desc]
    elif len(ftg) == 2:
        print('Looks like sites are encoded with two characters.')

        raise NotImplementedError('Cannot currently handle this encoding.')
    else:
        print('Each site seems to be encoded with more than two characters? Stopping here. \N{cross mark}')

if phenotype_path is not None:
    if args.phenotype_name not in list(pheno_raw.columns):
        print('Can\'t find the phenotype in the phenotype file by column name. These are the columns:')
        print(list(pheno_raw.columns))
        exit()

    print('Filtering the list of phenotypes to coerce missing ones to NaN...', end='', flush=True)
    filtered_phenos = list(pheno_raw[args.phenotype_name].apply(lambda x: pd.to_numeric(x, errors='coerce')))
    print('\N{heavy check mark}')

all_geno = []
all_pheno = []
all_taxa = []
missing_geno = 0
missing_pheno = 0

if phenotype_path is not None:
    print('Using all of the taxa names from the phenotype file.')
    available_taxa = list(pheno_raw.iloc[:, 0])
    phenos = filtered_phenos
else:
    print('Using all of the taxa from the genotype file.')
    available_taxa = list(geno_raw.columns)[first_col:]
    phenos = np.zeros_like(available_taxa)

print('Compiling data now, this may take a while...', end='', flush=True)

for taxa, pheno in zip(available_taxa, phenos):
    if taxa not in geno_raw.columns:
        missing_geno += 1
        continue

    if phenotype_path is not None and isnan(pheno):
        missing_pheno += 1
        continue

    geno = np.array([snp_dict[i][p] for i, p in enumerate(list(geno_raw[taxa]))])

    if np.any(np.isnan(geno)) and not args.ignore_missing:
        print('Found a missing marker for taxa {0}'.format(taxa))
        print('Missing data is not supported, please impute missing markers first. \N{cross mark}')
        print('(If you really want, you can force past this error by setting --ignore_missing to set missing values to '
              'homo major but it\'s not recommended.)')
        exit()

    all_geno.append(geno)
    all_taxa.append(taxa)

    if phenotype_path is not None:
        all_pheno.append(pheno)

print('\N{heavy check mark}')

print('Skipped due to missing genotype: {0}'.format(missing_geno))
print('Skipped due to missing phenotype: {0}'.format(missing_pheno))

if len(all_geno) == 0:
    print('All of the individuals were skipped, either because all the taxa ID were missing from the genotype file, or '
          'because the phenotypes could not be interpreted as numbers. \N{cross mark}')
    exit()

print('Total samples: {0}'.format(len(all_geno)))
print('Finishing up and writing to disk...', end='', flush=True)

all_geno = np.stack(all_geno).astype(np.int8)
all_pheno = np.array(all_pheno)
all_taxa = np.array(all_taxa)

output_filename = path.join(path.dirname(genotype_path), '{0}.variants2.bin'.format(args.phenotype_name if phenotype_path is not None else 'no-phenotypes'))
dump((all_geno, all_pheno, all_taxa), output_filename)

print('\N{heavy check mark}')

print('Output file was written at {0} \N{beer mug}'.format(output_filename))
