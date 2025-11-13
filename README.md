# Foldtree_ProstT5

<img src="logo.png" alt="Foldtree_ProstT5 Logo" height="800">

⚠️ **CAUTION: This pipeline has not been benchmarked yet. Use at your own risk and validate results carefully.**

A Snakemake pipeline that provides a Foldtree replacement for phylogenetic tree construction when protein structures are not available. This pipeline leverages ProstT5 embeddings through Foldseek to generate statistically corrected and rooted sequence identity trees.

## Overview

Foldtree_ProstT5 is designed for scenarios where:
- Protein structures are unavailable for your sequences of interest
- You need phylogenetic trees based on structural similarity estimates
- Traditional Foldtree cannot be used due to lack of structural data

The pipeline operates in "fident mode" (sequence identity mode) and provides:
- ✅ Statistically corrected sequence identity trees
- ✅ Rooted phylogenetic trees
- ❌ **Does NOT output** LDDT distance matrices
- ❌ **Does NOT output** TM-score distance matrices

## Prerequisites

### Required Software
- [Snakemake](https://snakemake.readthedocs.io/) (≥7.0)
- [Foldseek](https://github.com/steineggerlab/foldseek)
- [Conda/Mamba](https://conda.io/) for environment management

### Required Data
You **must** download the ProstT5 weights before running this pipeline:

```bash
# Download ProstT5 weights using Foldseek
foldseek databases ProstT5 weights tmp
```

**Important**: Ensure you have sufficient disk space (~XX GB) for the ProstT5 database.

## Directory Structure

```
├── Snakefile                 # Main workflow file
├── config/                   # Configuration files
│   ├── config.yaml          # Main configuration
│   ├── cluster_config.yaml  # Cluster-specific settings
│   └── profile_config.yaml  # Snakemake profile configuration
├── workflow/                 # Workflow components
│   ├── rules/               # Rule definitions
│   │   ├── common.smk       # Common functions and constraints
│   │   ├── preprocessing.smk # Sequence preprocessing
│   │   └── fold_tree_prostT5.smk # ProstT5-based tree construction
│   ├── scripts/             # Custom scripts
│   │   ├── prostT5_analysis.py # ProstT5 embedding analysis
│   │   └── tree_correction.py # Statistical correction and rooting
│   └── envs/                # Conda environment files
│       ├── foldseek.yaml    # Foldseek environment
│       └── phylo.yaml       # Phylogenetic tools environment
├── data/                    # Input data
│   ├── sequences/           # Input protein sequences (FASTA)
│   └── prostT5_weights/     # ProstT5 database weights
├── resources/               # Reference files and static resources
├── results/                 # Output files
│   ├── embeddings/          # ProstT5 embeddings
│   ├── distances/           # Sequence identity matrices
│   ├── trees/               # Generated phylogenetic trees
│   └── reports/             # Analysis reports
├── logs/                    # Log files
└── run_pipeline.sh          # Convenience script to run pipeline
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/DessimozLab/Foldtree_ProstT5.git
cd Foldtree_ProstT5

# Install Snakemake (if not already installed)
conda install -c bioconda snakemake

# Set up the project structure
make setup
```

### 2. Download ProstT5 Weights
```bash
# This may take several hours and requires significant disk space
foldseek databases ProstT5 data/prostT5_weights tmp
```

### 3. Prepare Input Data
Place your protein sequences in FASTA format in `data/sequences/`:
```bash
cp your_sequences.fasta data/sequences/
```

### 4. Configure the Pipeline
Edit `config/config.yaml` to specify:
```yaml
# Input sequences
input_sequences: "data/sequences/your_sequences.fasta"

# ProstT5 database path
prostT5_db: "data/prostT5_weights/ProstT5"

# Analysis parameters
min_sequence_length: 30
max_sequence_length: 1000
similarity_threshold: 0.3

# Tree construction parameters
correction_method: "jukes_cantor"  # Options: jukes_cantor, kimura
rooting_method: "midpoint"         # Options: midpoint, outgroup
```

### 5. Run the Pipeline
```bash
# Test the workflow (dry run)
./run_pipeline.sh --dry-run

# Run with 4 cores
./run_pipeline.sh --cores 4

# Or run directly with Snakemake
snakemake --use-conda --cores 4
```

## Output

The pipeline generates several key outputs in the `results/` directory:


## Pipeline Workflow

1. **Sequence Preprocessing**: Validates and filters input sequences
2. **ProstT5 Embedding**: Generates structural embeddings using Foldseek + ProstT5
3. **Distance Calculation**: Computes sequence identity from embeddings
4. **Tree Construction**: Builds initial phylogenetic tree from distance matrix
5. **Statistical Correction**: Applies evolutionary distance corrections
6. **Tree Rooting**: Roots the tree using specified method

## Limitations & Important Notes

⚠️ **Critical Limitations:**
- **No LDDT matrices**: This pipeline cannot output LDDT-based distance matrices
- **No TM-score matrices**: TM-score calculations are not supported
- **Fident mode only**: Only operates in sequence identity mode, not structural similarity mode
- **Requires ProstT5 weights**: Must download large database files (~XX GB)
- **Not benchmarked**: Results should be validated against known phylogenies when possible

## Usage Examples

### Basic Usage
```bash
# Run with default settings
snakemake --use-conda --cores 8 -s workflow/rules/fold_tree_prostT5 --config folder=./sequences 
````

### Cluster Execution
```bash
# For SLURM clusters
snakemake --cluster-config config/cluster_config.yaml \
          --cluster "sbatch --partition=normal --time=4:00:00" \
          --jobs 20 --use-conda --directory #your dataset path#
```

## Troubleshooting

### Common Issues
1. **ProstT5 database missing**: Ensure you've downloaded the weights using `foldseek databases`
2. **Memory errors**: Increase memory allocation in cluster config for large datasets
3. **Slow performance**: Use more cores or consider splitting large sequence sets

## Citation

If you use this pipeline, please cite:
- **Foldseek**: [Steinegger & Söding, 2022]
- **ProstT5**: [Heinzinger et al., 2023]
- **Foldtree** : [Moi et al., 2025]

## Contributing

This pipeline is under active development. Please report issues or contribute improvements via GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
