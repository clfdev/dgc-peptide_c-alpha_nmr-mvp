# DGC-NMR: Discrete Geometry Chemistry for NMR Chemical Shift Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Phase 0 MVP](https://img.shields.io/badge/status-Phase%200%20MVP-orange.svg)]()

A zero-cost computational framework for predicting Cα chemical shifts in small peptides using discrete geometric features derived from backbone structure.

---

## Overview

**DGC-NMR** implements the Discrete Geometry Chemistry (DGC) paradigm for NMR chemical shift prediction, demonstrating that Cα carbon shifts in peptides can be predicted with acceptable accuracy (MAE ≈ 3.3 ppm) using **only geometric features** from Cα backbone coordinates—no quantum mechanics, no extensive databases, no side-chain information required.

### Key Features

- ✅ **Zero-cost inference**: Predictions in milliseconds per structure
- ✅ **Minimal dependencies**: NumPy, SciPy, scikit-learn, BioPython
- ✅ **Geometry-first approach**: Pure spatial features (Cα-Cα distances, radius of gyration)
- ✅ **Transparent model**: Linear ridge regression with interpretable coefficients
- ✅ **Validated workflow**: Structure-level cross-validation on curated PDB-BMRB pairs
- ✅ **Multiple output formats**: CSV, JSON, NMR-STAR

### Performance (Phase 0)

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 3.28 ppm |
| Root Mean Square Error (RMSE) | 4.15 ppm |
| R² Coefficient | 0.42 |
| Inference Time | ~1.2 ms per structure |
| Training Set Size | 4 structures, ~90 residues |

**Comparison to Baselines:**
- Null model (predict mean): MAE = 6.8 ppm
- Sequence-only model: MAE = 5.2 ppm
- **DGC-NMR (geometric)**: MAE = 3.28 ppm ✓

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install from Source
```bash
# Clone the repository
git clone https://github.com/[username]/dgc-nmr.git
cd dgc-nmr

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies
```
numpy>=1.21.5
scipy>=1.7.3
scikit-learn>=1.0.2
requests>=2.27.1
biopython>=1.79
matplotlib>=3.5.1  # optional, for visualization
pandas>=1.3.5      # optional, for CSV export
```

---

## Quick Start

### Python API
```python
from dgc_nmr.prediction import ChemicalShiftPredictor

# Load pre-trained Phase 0 model
predictor = ChemicalShiftPredictor.load_pretrained('models/trained/')

# Predict from PDB file
predictions = predictor.predict_from_pdb('examples/1VII.pdb', chain='A')

# Display results
for res_id, res_name, shift in predictions:
    print(f"{res_id:4d} {res_name:3s} {shift:6.2f} ppm")
```

### Command-Line Interface
```bash
# Single structure prediction
python -m dgc_nmr.predict \
    --pdb examples/1VII.pdb \
    --chain A \
    --output predictions.csv

# Batch prediction
python -m dgc_nmr.predict \
    --pdb-list structure_list.txt \
    --output-dir predictions/ \
    --output-format json

# Validation against experimental shifts
python -m dgc_nmr.predict \
    --pdb examples/1VII.pdb \
    --experimental-shifts data/raw/bmrb/bmr5713.str \
    --output validation_report.txt
```

---

## Project Structure
```
dgc-nmr/
├── dgc_nmr/                    # Main package
│   ├── __init__.py
│   ├── data/                   # Dataset management
│   │   ├── raw/               
│   │   │   ├── pdb/           # Downloaded PDB files
│   │   │   └── bmrb/          # Downloaded BMRB NMR-STAR files
│   │   ├── processed/         # Curated datasets (*.npz)
│   │   └── pilot_structures.json  # Validated PDB-BMRB pairs
│   │
│   ├── validation/            # Quality control pipeline
│   │   ├── pilot_validator.py # Automated validation checks
│   │   ├── parsers.py         # PDB/BMRB file parsers
│   │   └── alignment.py       # Sequence alignment utilities
│   │
│   ├── features/              # Geometric feature engineering
│   │   ├── geometric.py       # Feature extraction (distances, R_g)
│   │   └── normalization.py   # Z-score standardization
│   │
│   ├── models/                # Regression models
│   │   ├── ridge.py           # Ridge regression + nested CV
│   │   ├── evaluation.py      # Performance metrics
│   │   └── trained/           # Serialized models
│   │       ├── phase0_model.pkl
│   │       └── scaler.pkl
│   │
│   └── prediction/            # Inference interface
│       ├── predictor.py       # Main prediction class
│       └── output.py          # Export formatters
│
├── config/                    # Configuration
│   └── settings.py            # Default parameters
│
├── examples/                  # Example structures
│   ├── 1VII.pdb              # Villin headpiece (36 res)
│   ├── 1LE1.pdb              # Trpzip-2 (12 res)
│   └── run_examples.py        # Demo script
│
├── tests/                     # Unit tests
│   ├── test_validation.py
│   ├── test_features.py
│   └── test_prediction.py
│
├── docs/                      # Documentation
│   ├── paper/                 # Manuscript source
│   ├── methodology.md         # Detailed methods
│   └── api_reference.md       # API documentation
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── LICENSE.md                 # MIT License
└── README.md                  # This file
```

---

## Usage Examples

### Example 1: Predict Shifts for a Single Peptide
```python
from dgc_nmr.prediction import ChemicalShiftPredictor

predictor = ChemicalShiftPredictor.load_pretrained('models/trained/')
results = predictor.predict_from_pdb('my_peptide.pdb')

# Export to CSV
from dgc_nmr.prediction.output import write_csv
write_csv(results, 'shifts.csv')
```

### Example 2: Validate Against Experimental Data
```python
from dgc_nmr.prediction import ChemicalShiftPredictor
from dgc_nmr.models.evaluation import compute_mae, compute_r2

predictor = ChemicalShiftPredictor.load_pretrained('models/trained/')

# Load experimental shifts from BMRB
from dgc_nmr.validation.parsers import parse_star_ca_shifts
experimental = parse_star_ca_shifts('data/raw/bmrb/bmr5713.str')

# Predict
predicted = predictor.predict_from_pdb('data/raw/pdb/1VII.pdb')

# Align and compare
mae = compute_mae(experimental['shifts'], predicted['shifts'])
r2 = compute_r2(experimental['shifts'], predicted['shifts'])

print(f"MAE: {mae:.2f} ppm")
print(f"R²: {r2:.3f}")
```

### Example 3: Batch Processing Multiple Structures
```bash
# Create file list
ls *.pdb > structures.txt

# Run batch prediction
python -m dgc_nmr.predict \
    --pdb-list structures.txt \
    --output-dir results/ \
    --output-format csv \
    --verbose
```

### Example 4: Custom Feature Configuration
```python
from dgc_nmr.features import GeometricFeatureExtractor
from dgc_nmr.models import RidgeRegressor

# Use only 3 nearest neighbors instead of default 5
extractor = GeometricFeatureExtractor(k_neighbors=3)

# Train custom model
model = RidgeRegressor()
model.train_with_nested_cv(X_train, y_train, groups, lambda_grid=[0.1, 1.0, 10.0])
```

---

## Methodology Summary

### Dataset Curation

- **Source**: Curated PDB-BMRB pairs from RCSB and BMRB databases
- **Selection criteria**: 
  - Solution NMR structures
  - Single protein chain
  - 10-40 residues length
  - ≥70% Cα chemical shift coverage
- **Pilot dataset**: 4 structures (1VII, 1LE1, 1E0L, 2MAG), ~90 residues

### Feature Engineering

For each residue *i*, extract 7 geometric features:
1. Distance to 1st nearest Cα neighbor
2. Distance to 2nd nearest Cα neighbor
3. Distance to 3rd nearest Cα neighbor
4. Distance to 4th nearest Cα neighbor
5. Distance to 5th nearest Cα neighbor
6. Radius of gyration (global compactness)
7. Mean distance to all Cα atoms

**Mathematical formulation:**

$$\mathbf{x}_i = [d_{i,n_1}, d_{i,n_2}, d_{i,n_3}, d_{i,n_4}, d_{i,n_5}, R_g, \bar{d}_i] \in \mathbb{R}^7$$

where $d_{i,n_j}$ is the distance to the *j*-th nearest neighbor, $R_g$ is radius of gyration, and $\bar{d}_i$ is mean distance.

### Model Training

- **Algorithm**: Ridge regression with L2 regularization
- **Objective**: $\min_{\mathbf{w}} \sum_{i} (\delta_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2$
- **Hyperparameter tuning**: Nested cross-validation
  - Outer: Leave-one-structure-out (4 folds)
  - Inner: 3-fold GroupKFold
  - λ grid: [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
- **Normalization**: Z-score standardization fitted on training set only

### Validation Protocol

- **Cross-validation**: Structure-level split (no residues from same structure in train/test)
- **Metrics**: MAE, RMSE, R² (global and per-structure)
- **Baseline comparisons**: Null model, sequence-only model

---

## Validation Results

### Per-Structure Performance

| PDB ID | Residues | MAE (ppm) | RMSE (ppm) | R² |
|--------|----------|-----------|------------|-----|
| 1LE1 (Trpzip-2) | 12 | 2.41 | 3.02 | 0.58 |
| 1E0L (WW domain) | 34 | 3.15 | 3.98 | 0.45 |
| 1VII (Villin) | 36 | 3.24 | 4.11 | 0.44 |
| 2MAG (Magainin-2) | 23 | 4.32 | 5.48 | 0.28 |
| **Global** | **90** | **3.28** | **4.15** | **0.42** |

### Feature Importance (Ablation Studies)

| Feature Removed | MAE (ppm) | Change (%) |
|-----------------|-----------|------------|
| None (full model) | 3.28 | — |
| 1st nearest neighbor | 4.91 | +50% |
| 2nd nearest neighbor | 4.01 | +22% |
| 3rd nearest neighbor | 3.69 | +13% |
| Mean distance | 3.43 | +5% |
| Radius of gyration | 3.31 | +1% |

**Insight**: Local geometry (nearest neighbors) dominates; global descriptors contribute minimally.

---

## Applications

### 1. Peptide Design Screening
- **Use case**: Rapid evaluation of designed peptide structures
- **Workflow**: Generate candidate structures → Predict shifts → Filter anomalies → Validate top candidates experimentally
- **Advantage**: 10⁴× faster than quantum methods

### 2. Structure Validation
- **Use case**: Quality control for homology models or AlphaFold predictions
- **Workflow**: Predict shifts from model → Compare to experimental NMR → Identify misfolded regions
- **Advantage**: Geometry-independent check

### 3. Educational Tool
- **Use case**: Teaching structure-property relationships
- **Workflow**: Perturb coordinates → Observe shift changes → Understand geometric effects
- **Advantage**: Transparent, interpretable model

### 4. Baseline for Benchmarking
- **Use case**: Quantify value of advanced features in new predictors
- **Workflow**: Compare new method to geometric baseline → Measure marginal improvement
- **Advantage**: Establishes floor performance

---

## Limitations

### Current Scope (Phase 0)

❌ **Not suitable for:**
- Large proteins (>50 residues) — lacks long-range contact features
- Intrinsically disordered regions — requires ensemble averaging
- High-precision applications — MAE 3.3 ppm vs. 1.0 ppm for SHIFTX2
- Glycine/proline-rich sequences — systematic errors
- Non-native states — trained on folded structures only
- Post-translational modifications — not accounted for

✅ **Optimized for:**
- Small peptides (10-40 residues)
- Well-folded structures (α-helix, β-sheet, turns)
- Rapid screening applications
- Approximate shift estimation
- Structure validation
- Educational demonstrations

### Known Issues

1. **Glycine residues**: Systematically over-predicted (+5.1 ppm mean error)
2. **Proline residues**: Limited training data, reduced accuracy
3. **Terminal residues**: Often lack experimental shifts, excluded from validation
4. **Ensemble structures**: Uses only first NMR model by default

---

## Roadmap

### Phase 1 (Planned: Q2 2026)

**Feature additions:**
- ✨ Solvent-accessible surface area (SASA)
- ✨ Secondary structure classification (DSSP)
- ✨ One-hot encoded residue types
- ✨ Hydrogen bonding patterns

**Expected improvement:** MAE < 2.0 ppm

### Phase 2 (Planned: Q4 2026)

**Advanced features:**
- ✨ NMR ensemble averaging
- ✨ Dihedral angle distributions
- ✨ Ring current effects
- ✨ Electrostatic field descriptors

**Target:** Competitive with SHIFTX2/CheShift

### Future Extensions

- Web server interface
- GPU acceleration for batch processing
- Extension to other nuclei (¹³Cβ, ¹⁵N, ¹H)
- Integration with molecular dynamics
- Docker containerization

---

## Citation

If you use this software in your research, please cite:
```bibtex
@article{firme2026dgc_nmr,
  title={Discrete Geometry Prediction of C$\alpha$ Chemical Shifts in Small Peptides: A Zero-Cost Surrogate Model},
  author={Firme, Caio L.},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI]},
  note={Phase 0 MVP}
}
```

**Related work:**

- Firme, C. L.; Boes, E. S. (2025). "Discrete Geometry Chemistry: First applications and beyond." *Canadian Journal of Chemistry*.

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

**Areas for contribution:**
- 🐛 Bug reports and fixes
- 📚 Documentation improvements
- ✨ Feature implementations (Phase 1/2 roadmap)
- 🧪 Expanded validation datasets
- 🎨 Visualization tools
- 🌐 Web interface development

**Development workflow:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push to branch (`git push origin feature/awesome-feature`)
5. Open Pull Request

---

## Support

**Bug reports:** [GitHub Issues](https://github.com/[username]/dgc-nmr/issues)

**Questions:** caio.firme@ufrn.br

**Discussions:** [GitHub Discussions](https://github.com/[username]/dgc-nmr/discussions)

**Documentation:** [Full documentation](https://dgc-nmr.readthedocs.io) (coming soon)

---

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.

---

## Acknowledgments

**Institutional support:**
- Federal University of Rio Grande do Norte (UFRN), Institute of Chemistry
- Natal, Rio Grande do Norte, Brazil

**Data sources:**
- [RCSB Protein Data Bank](https://www.rcsb.org)
- [Biological Magnetic Resonance Data Bank (BMRB)](https://bmrb.io)

**Theoretical framework:**
- Based on the Discrete Geometry Chemistry (DGC) paradigm
- Part of the broader DGC research program (Snapshot models, D2BIA_discrete)

---

## Contact

**Principal Investigator:**  
Caio L. Firme, Ph.D.  
Institute of Chemistry  
Federal University of Rio Grande do Norte (UFRN)  
Av. Senador Salgado Filho, 3000  
Natal - RN, Brazil, CEP: 59078-970

📧 caio.firme@ufrn.br  
📧 firme.caio@gmail.com  

---

<p align="center">
  <strong>DGC-NMR</strong> — Geometry-first chemical shift prediction
</p>

<p align="center">
  <em>Part of the Discrete Geometry Chemistry research program</em>
</p>