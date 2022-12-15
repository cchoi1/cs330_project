#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="cifar-oe-tune" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ./CIFAR

python oe_tune.py --method oe --id_dataset cifar10 --ood_dataset cifar100 --model wrn
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset cifar100 --model wrn
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset cifar100 --model wrn --use_test_ood
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset cifar100 --model wrn --use_test_ood --misclassified_id

python oe_tune.py --method oe --id_dataset cifar10 --ood_dataset svhn --model wrn
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset svhn --model wrn
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset svhn --model wrn --use_test_ood
python oe_tune.py --method ours --id_dataset cifar10 --ood_dataset svhn --model wrn --use_test_ood --misclassified_id

python oe_tune.py --method oe --id_dataset cifar100 --ood_dataset cifar10 --model wrn
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset cifar10 --model wrn
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset cifar10 --model wrn --use_test_ood
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset cifar10 --model wrn --use_test_ood --misclassified_id

python oe_tune.py --method oe --id_dataset cifar100 --ood_dataset svhn --model wrn
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset svhn --model wrn
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset svhn --model wrn --use_test_ood
python oe_tune.py --method ours --id_dataset cifar100 --ood_dataset svhn --model wrn --use_test_ood --misclassified_id