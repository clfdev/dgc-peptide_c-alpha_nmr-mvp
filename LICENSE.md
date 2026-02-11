# MIT License

Copyright (c) 2026 Caio L. Firme - Federal University of Rio Grande do Norte (UFRN)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use this software in your research, please cite:

**Firme, C. L.** (2026). *Discrete Geometry Prediction of Cα Chemical Shifts in Small Peptides: A Zero-Cost Surrogate Model*. [Journal Name], [Volume], [Pages]. DOI: [to be assigned]

BibTeX entry:
```bibtex
@article{firme2026dgc_nmr,
  title={Discrete Geometry Prediction of C$\alpha$ Chemical Shifts in Small Peptides: A Zero-Cost Surrogate Model},
  author={Firme, Caio L.},
  journal={[Journal Name]},
  volume={[Volume]},
  pages={[Pages]},
  year={2026},
  doi={[DOI]},
  publisher={[Publisher]}
}
```

---

## Acknowledgments

This work is part of the **Discrete Geometry Chemistry (DGC)** research program at the Federal University of Rio Grande do Norte (UFRN), Institute of Chemistry.

**Data Sources:**
- Protein Data Bank (PDB): https://www.rcsb.org
- Biological Magnetic Resonance Data Bank (BMRB): https://bmrb.io

**Related Publications:**
1. Firme, C. L.; Boes, E. S. (2025). "Discrete Geometry Chemistry: First applications and beyond." *Canadian Journal of Chemistry*.

2. Firme, C. L. (2026). "The 1D Snapshot Model for Conjugated Linear Chains." [Manuscript submitted for publication].

3. Firme, C. L. (2026). "The 2D Snapshot Model for Polycyclic Aromatic Hydrocarbons." [Manuscript submitted for publication].

---

## Disclaimer

This software is provided for **research and educational purposes only**. The chemical shift predictions are approximate and should not be used as a substitute for experimental NMR measurements or high-accuracy quantum chemical calculations in critical applications such as:

- Clinical diagnostics
- Pharmaceutical quality control
- Regulatory submissions
- High-stakes structure validation

Users are responsible for validating predictions against experimental data before drawing scientific conclusions or making decisions based on the software output.

**Known Limitations:**
- Optimized for small peptides (10-40 residues)
- Mean Absolute Error (MAE) ≈ 3.3 ppm on validation set
- Systematic errors for glycine and proline residues
- Not validated on large proteins, disordered regions, or non-native states
- Does not account for post-translational modifications or ligand binding

For production applications requiring higher accuracy, consider established predictors such as SHIFTX2, CheShift, or SPARTA+.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/improvement`)
3. **Commit** your changes with clear messages
4. **Push** to your fork (`git push origin feature/improvement`)
5. Open a **Pull Request** with detailed description

**Code Standards:**
- Follow PEP 8 style guidelines
- Include docstrings for all public functions and classes
- Add unit tests for new features
- Update documentation as needed

**Bug Reports:**
Please open an issue including:
- Minimal reproducible example
- Python version and dependency versions
- Expected vs. actual behavior
- Error messages or stack traces

---

## Contact

**Principal Investigator:**  
Caio L. Firme, Ph.D.  
Institute of Chemistry  
Federal University of Rio Grande do Norte (UFRN)  
Av. Senador Salgado Filho, 3000  
Natal - RN, Brazil, CEP: 59078-970  

Email: caio.firme@ufrn.br  
Alternative: firme.caio@gmail.com  
Phone: +55 (84) 3342-2323 Ext. 144  

**Research Group Website:** [To be added]  
**GitHub Repository:** https://github.com/[username]/dgc-nmr  

---

## Version History

**v0.1.0** (February 2026) - Phase 0 MVP Release
- Initial implementation of geometric feature extraction
- Ridge regression model with nested cross-validation
- Pilot dataset validation (4 structures)
- Command-line and Python API interfaces
- Basic documentation and examples

**Planned Releases:**
- **v0.2.0** (Phase 1): Addition of SASA, DSSP, residue one-hot encoding
- **v1.0.0**: Production-ready release with expanded dataset (20+ structures)

---

## License Compatibility

This software uses the following open-source libraries, each with their own licenses:

- **NumPy** (BSD License): https://numpy.org/doc/stable/license.html
- **SciPy** (BSD License): https://scipy.org/scipylib/license.html
- **scikit-learn** (BSD License): https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
- **Requests** (Apache 2.0): https://github.com/psf/requests/blob/main/LICENSE
- **BioPython** (Biopython License): https://github.com/biopython/biopython/blob/master/LICENSE.rst

All dependencies are compatible with the MIT License used for this project.

---

*Last updated: February 11, 2026*