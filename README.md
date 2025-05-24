# Project Setup Instructions

This readme is for CIKM submission: Joint Learning of Drug-Drug Combination and Drug-Drug Interaction via Coupled Tensor-Tensor Factorization with Side

This guide explains how to set up the Python environment required to run the project, and how to execute the SI-ADMM algorithm along with the baseline models.

## Requirements

- Python **3.7**
- `pip` (Python package manager)

## Environment Setup

### 1. Create and Activate a Virtual Environment (Recommended)

Ensure you are in the root directory of the `Joint_Admm` repository.

Using `venv`:

```bash
python3.7 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using `conda`:

```bash
conda create -n myenv python=3.7
conda activate myenv
```

### 2. Install Required Packages

Install all necessary dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Confirm that all key packages are installed correctly:

```bash
python -c "import numpy, pandas, sklearn, scipy, matplotlib, tensorly; print('Environment ready!')"
```

## Running the Project

### Preprocessing Data

Before running the main algorithm or any baseline model, navigate to the `src` directory and preprocess the data to generate the tensors:

```bash
python data_preprocessing.py
```

This will create new tensor files in both the `./data/` and `./baseline/ADMM/` directories.

### Running SI-ADMM

To execute the SI-ADMM algorithm, use a command like the following from within the `src` directory:

```bash
python analytic_approach.py --save_name "my_run_" --x_data cdcdb --y_data tdc --rank 60 --rnd_seeds 123 --tolerance 2 --x_weight 0.57642897 --y_weight 0.04189498 --si_weight 0.08469067 0.09330971 0.02414446 0.17953121
```

Results will be saved in the `./output/` directory.

You can explore additional parameter options in the `parse_arguments()` function, starting at line 177 in `analytic_approach.py`.

### Running Baseline Models

To run baseline models, navigate to the `./baseline/` directory and execute:

```bash
python experiment.py --data x --model CP CTF --rank 4
```

- The `--data` argument accepts `x` or `y`, corresponding to the X tensor or Y tensor prediction tasks.
- The `--model` argument accepts one or all or part of the following: `CTF`, `TFAI_CP_within_mod`, `TDRC`, or `CP`.
- The`--rank` is the number of rank of the decomposed latent matrix.

Results from the baseline runs will be saved in the `./baseline/baseline_output/` directory.

## Notes

- Python 3.7 is required to ensure compatibility with all specified dependencies.
- If you intend to use Jupyter notebooks or additional development tools, you may need to install them separately.
