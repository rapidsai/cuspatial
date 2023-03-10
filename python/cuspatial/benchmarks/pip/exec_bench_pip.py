import subprocess

columns = ["mean", "stddev", "rounds"]
cmd = [
    "pytest",
    "bench_pip.py",
    f'--benchmark-columns={",".join(columns)}',
    "-x",
    "--pdb",
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, error = process.communicate()
print(output)
print(error)
