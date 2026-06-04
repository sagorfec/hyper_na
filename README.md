# Physical Mechanisms and Fundamental Limits of High-NA EUV Lithography - Multi Physics Simulation Framework ( NA = 0.55 , HP = 8 nm)

### Short description: 
This repository contains the open-source code, simulation inputs, validation suite, and the datasets produced for the Elsevier paper "Physical Mechanisms and Fundamental Limits of High-NA EUV Lithography: A Multi-Physics Simulation Framework at the 8 nm Half-Pitch Node". The package implements a fully-coupled GPU-accelerated simulation stack that integrates multilayer TMM reflectivity, partially coherent anamorphic aerial-image generation, Richards-Wolf vector polarization, a four-stage resist model (SE blur, Dill Beer-Lambert exposure, PEB diffusion, Mack-4 dissolution), Monte Carlo shot-noise realizations, and post-processing analyses (NILS, LWR, CDE, Shannon entropy, RLS triangle).

## Repository Structure

```
hyper_na/
├── scripts/
│   ├── High_NA.py                   # Single run script Fast if you run it in Kaggel
├── results/                         # CSV results
├── figures/                         # Output figures (PDF)

```


## Reproducibility

All results in the paper can be reproduced by running the scripts in order.


## Citation

```bibtex
@article{sagor2026hyper_na,
  title   = {Physical Mechanisms and Fundamental Limits of High-NA EUV Lithography: 
  A Multi-Physics Simulation Framework at the 8\,nm Half-Pitch Node},
  author  = { Md. Ifthakhar Khan Sagor, Md. Sanawar Hossain, Md. Zillur Rahaman, Partha Mandal, Anit Barua},
  journal = {Elsevier},
  year    = {2026},
  note    = {Under review}
}
```
---

## License

MIT License. See LICENSE file.

