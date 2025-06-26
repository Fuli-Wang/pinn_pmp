# pinn_pmp
Passive motion paradigm for parallel robots using self-supervised physics-informed neural networks

# PINN training
To implement the paradigm, the fist work is training the PINN model. 

We provided some MATLAB code in visualization folder where you can define your platform and visualize it.  

    visualization.m

![platform](https://github.com/Fuli-Wang/pinn_pmp/blob/visualization/platform.png)
    
To inverse kinematics is well-defined (it can generate data for ANN and PINN training):

    data.m
  
Once data is generated, you can trian a PINN model:

    python3 pinntrain.py

# PMP implementation

After the training, you will get a model(.pth), then set a target length (.txt) and run the following example to excute PMP which will print and record the result (.txt)

    python3 pmp_parallel.py
