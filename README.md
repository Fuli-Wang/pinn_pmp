This work has since been substantially reformulated, standardised, and extended into **KS-MP (Kinematic–Synergy Motor Primitives)**, a unified motor-primitive framework designed to support control across different robot morphologies.

The latest implementation is available here:

**[KS-MP: Kinematic–Synergy Motor Primitives](https://github.com/Fuli-Wang/KS-MP)**

Compared with KS-MP, the implementation in this repository is less standardised in its control formulation, software structure, cross-morphology applicability, and overall control performance. It therefore no longer represents the current version of our framework and **will not receive further updates or maintenance**.


# pinn_pmp
Passive motion paradigm for parallel robots using self-supervised physics-informed neural networks

# PINN training
To implement the paradigm, the first step is to train the PINN model. 

We provided some MATLAB code in the visualization folder, where user can reuse and define their platform and visualize it.  

    visualization.m

![platform](visualization/platform.png)
    
Inverse kinematics is well-defined in the data file (it can generate data for ANN and PINN training):

    data.m
  
Once data is generated, you can train a PINN model:

    python3 pinntrain.py

Alternatively, users may directly employ a Python file to generate data and conduct training. The following is an example for a delta robot:

    python3 pinntrain_delta.py

# PMP implementation

After the training, a model(.pth) will be obtained, then please set your target length (.txt) and run the following example to execute PMP, which will print and record the result (.txt)

    python3 pmp_parallel.py

Back in the visualization folder, the results can be plotted:

    plot_result.m

![result](visualization/result.png)

## Citation

If this work is helpful, please cite:

```bibtex
@article{wang2026physicsinformed,
  author  = {Fuli Wang and Fazair Nizar Siraj and Windo Hutabarat and Ashutosh Tiwari},
  title   = {Physics-Informed Passive Motion Paradigm for Parallel Robots:
             A High-Precision Motor-Primitives Framework},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11},
  number  = {2},
  pages   = {1874--1881},
  year    = {2026},
  doi     = {10.1109/LRA.2025.3645663}
}
```
