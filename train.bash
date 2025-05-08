eval "$(/home/andrew/miniconda3/bin/conda shell.bash hook)"

conda activate snorm
config_name=mlp_2_512
python trainer.py --config configs/$config_name.yaml --output_dir runs/$config_name