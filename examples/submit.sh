#!/bin/bash

# List of filenames
files=(
	"h2_static_dynamic.py"
	"iron_bulk_modulus.py"
	"iron_harmonic_fcc_bcc.py"
	"water_cp2k_noise.py"
    "water_path_integral_md.py"
    "water_train_validate.py"
    "alanine_replica_exchange.py"
)

curl -O https://raw.githubusercontent.com/molmod/psiflow/main/configs/lumi.yaml

run_dir=$(pwd)/run_examples
mkdir $run_dir && cp lumi.yaml $run_dir && cd $run_dir

# Loop over each filename
for filename in "${files[@]}"
do
    name="${filename%.*}"
    mkdir $name
    cp lumi.yaml $name

    cat > $name/job.sh <<EOF
#!/bin/bash
#
#SBATCH -p small
#SBATCH --account=project_465000847
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=$name
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -e error.txt
#SBATCH -o output.txt


curl -LJO https://github.com/molmod/psiflow/archive/main.zip
unzip -j psiflow-main.zip "psiflow-main/examples/data/*" -d data

curl -O https://raw.githubusercontent.com/molmod/psiflow/main/examples/$filename
python $filename lumi.yaml
EOF

    cd $name
    sbatch job.sh
    cd $run_dir

done
