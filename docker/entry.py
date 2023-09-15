import sys
import subprocess
import torch

print('Running in docker detected.')
print('CUDA {0} available.'.format('is' if torch.cuda.is_available() else 'is not'))
print()

if '--genotype_file' in sys.argv:
    cmd = ['python', '/app/parsing/variants2bin.py', '--docker'] + sys.argv[1:]
    print('Calling the variants2bin script like this:')
else:
    cmd = ['python', '/app/evaluate/inference.py', '--docker'] + sys.argv[1:]
    print('Calling the inference script like this:')

print(' '.join(cmd))

try:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while p.poll() is None:
        l = p.stdout.readline().decode("utf-8")
        if l != '':
            print(l)
except subprocess.CalledProcessError as exc:
    print('Something went wrong:', exc.returncode, exc.output)
