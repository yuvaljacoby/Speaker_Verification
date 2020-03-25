from datetime import datetime
import os
import sys


BASE_FOLDER = "/home/yuval/projects/Marabou/"
if os.path.exists("/cs/usr/yuvalja/projects/Marabou"):
    BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"

LOG_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/out_train/")
MODELS_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/models/")


def create_sbatch(output_folder, cache_folder=''):
    print("*" * 100)
    print("creating sbatch {}".format('using cache {}'.format(cache_folder) if cache_folder else ''))
    print("*" * 100)

    # if cache_folder:
    #     shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=1)
    for i in range(17,30):
        exp_time = str(datetime.now()).replace(" ", "-")

        # if check_if_model_in_dir(model, cache_folder):
        #     continue
        with open(os.path.join(output_folder, "train_config_{}.sh".format(i)), "w") as slurm_file:
            exp = "iterations".format()
            # model_name = model[:model.rfind('.')]
            slurm_file.write('#!/bin/bash\n')
            slurm_file.write('#SBATCH --job-name=train_sbatch_{}_{}\n'.format(i, exp_time))
            slurm_file.write('#SBATCH --cpus-per-task=10\n')
            slurm_file.write('#SBATCH --output={}train_{}.out\n'.format(LOG_FOLDER, i))
            slurm_file.write('#SBATCH --time=05:00:00\n')
            slurm_file.write('#SBATCH --mem-per-cpu=1020\n')
            slurm_file.write('#SBATCH --mail-type=END,FAIL\n')
            slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')

            slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Speaker_Verification\n')
            slurm_file.write('source /cs/labs/guykatz/yuvalja/tensorflow/bin/activate.csh\n')
            slurm_file.write('python3 spekaer_identification/model.py {}\n'.format(i))

if __name__ == "__main__":
    create_sbatch(sys.argv[1])
