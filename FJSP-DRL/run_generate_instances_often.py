import subprocess

for data_type in ["test", "vali"]:
    for variance in ["random", "fixed"]:
        for n_j, n_m in [(10, 5), (15, 10), (20, 5), (20, 10), (30, 10), (40, 10)]:
            subprocess.run(
                [
                    "python",
                    "generate_instances_with_realizations.py",
                    "--data_type",
                    data_type,
                    "--variance",
                    variance,
                    "--n_j",
                    str(n_j),
                    "--n_m",
                    str(n_m),
                ]
            )
