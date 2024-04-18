#!/bin/bash
#SBATCH -J nerf2_test                           # Job name
#SBATCH --account=gts-rs275                     # charge account
#SBATCH -N 1 --gres=gpu:V100:1                  # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=10G                       # Memory per core
#SBATCH -t 1000                                 # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                              # QOS Name
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=cchen847@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
pip install scipy==1.10.0
pip install mne==1.6.1
pip install mne_bids
pip install pyriemann
pip install scikit-learn
pip install moabb
python mDA_Physionet_hands_feet_MI.py --download_path ~/p-rs275-0/da_eeg/mDA/mne-data/
