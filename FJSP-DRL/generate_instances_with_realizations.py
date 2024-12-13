import os

import numpy as np
from tqdm import tqdm

from data_utils import (
    generate_data_to_files,
    load_data_from_single_file,
    matrix_to_text,
)
from generate_random_instances import (
    create_stochastic_realizations,
    create_stochastic_realizations_random_variances,
    get_all_files,
)
from params import configs

np.random.seed(configs.seed_datagen)

NUM_SAMPLES = 5000

print("Trying to generate instances")
if configs.data_type == "test":
    generate_data_to_files(
        configs.seed_datagen, f"./data/{configs.data_source}/", configs
    )
elif configs.data_type == "vali":
    generate_data_to_files(
        configs.seed_train_vali_datagen,
        f"./data/data_train_vali/{configs.data_source}/",
        configs,
    )
else:
    raise ValueError(
        f"Error from Instance Generation: incorrect data type {configs.data_type}"
    )

print("Trying to generate realizations")
train_vali_str = "/data_train_vali" if configs.data_type == "vali" else ""
directory = f"./data{train_vali_str}/{configs.data_source}/{configs.n_j}x{configs.n_m}+{configs.data_suffix}"
for original_instance in tqdm(get_all_files(directory), total=100):
    job_length, processing_times = load_data_from_single_file(original_instance)
    if configs.variance == "random":
        processing_times = create_stochastic_realizations_random_variances(
            processing_times, NUM_SAMPLES
        )
    elif configs.variance == "fixed":
        processing_times = create_stochastic_realizations(processing_times, NUM_SAMPLES)

    or_instance_name = original_instance.split("\\")[-1].split(".")[0]
    realization_file = f"./data{train_vali_str}/stochastic_realizations/{configs.variance}/{configs.data_source}/{configs.n_j}x{configs.n_m}+{configs.data_suffix}/{or_instance_name}.fjs"
    realization_dir = realization_file.removesuffix(f"/{or_instance_name}.fjs")
    if not os.path.exists(realization_dir):
        os.makedirs(realization_dir)
    with open(original_instance, "r") as f_or, open(realization_file, "w") as f_real:
        for line in f_or:
            f_real.write(line)
        for i in range(NUM_SAMPLES):
            f_real.write("\n")
            pt = processing_times[i]
            lines_doc = matrix_to_text(job_length, pt, "", first_line=False)
            for j in range(len(lines_doc)):
                f_real.write(lines_doc[j] + "\n")
