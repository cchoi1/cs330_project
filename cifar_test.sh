#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="cifar-test" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ./CIFAR

python test.py --num_to_avg 10 --method_name cifar10_cifar100_wrn_oe_tune
python test.py --num_to_avg 10 --method_name cifar10_svhn_wrn_oe_tune
python test.py --num_to_avg 10 --method_name cifar100_cifar10_wrn_oe_tune
python test.py --num_to_avg 10 --method_name cifar100_svhn_wrn_oe_tune