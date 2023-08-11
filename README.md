# FEniCSx scripts for MEALOR II Summer School

![test](https://mealor2.sciencesconf.org/data/pages/montage1.png)

Summer school website: https://mealor2.sciencesconf.org

## Installation instructions

1. Install Docker and Paraview on your system

2. Once Docker is installed, open a terminal and run the following commands

- 2.1 First, clone the MEALOR project repository
  
  `git clone https://github.com/MEALORII/TP_MEALOR.git`
  

- 2.2 Then pull the Docker image:
  
  - on Windows:
    
    `docker pull ghcr.io/bleyerj/mealor:latest`
    `docker run --init --rm -ti -p 8888:8888 --name mealor -e JUPYTER_ENABLE_LAB=yes -e CHOWN_HOME=yes -e CHOWN_EXTRAOPTS='-hR' --user root -v "%cd%":/home/jovyan/shared mealor:latest`
    

   - on Mac/Linux:
  
  `docker pull ghcr.io/bleyerj/mealor:latest`
  
  `docker run --init --rm -ti -p 8888:8888 --name mealor -e JUPYTER_ENABLE_LAB=yes -e CHOWN_HOME=yes -e CHOWN_EXTRAOPTS='-hR' --user root -v "$(pwd)":/home/jovyan/shared mealor:latest`
  

3. Then you should see something like:

```
To access the server, open this file in a browser:
 file:///home/jovyan/.local/share/jupyter/runtime/jpserver-369-open.html
 Or copy and paste one of these URLs:
 http://6af88686bce5:8888/lab?token=b955811ee149cada75db27258e6889a9cbee18a8afaeb373
 or http://127.0.0.1:8888/lab?token=b955811ee149cada75db27258e6889a9cbee18a8afaeb373
```

Click on one of the links or copy and paste into your Web browser

4. A JupyterLab instance should now open.
   Navigate into the `TP` folder
   Inside JupyterLab open a terminal and run
   `pip install mealor/`

5. You can now open any of the notebooks and start working.

Your results will be saved into ".xdmf" file formats that you can open with Paraview for visualization


### Requirements

* `FEniCSx` suite:  `dolfinx` >= 0.6
* `mfront` (TFEL 4.1) with Python bindings
* `MFrontGenericInterfaceSupport` (>= 2.1) with Python bindings
* `pyvista`
* `gmsh`

## Practical works

### [TP3 : Linear Elastic Fracture Mechanics](TP3_LEFM/LEFM.ipynb)

### Ductile fracture (GTN model)

- NT sample, influence of notch radius

- mesh size dependence
* anisotropic behaviour

### [TP6 : Implementation of damage gradient/phase-field models for brittle fracture](TP6_Variational_damage_gradient/Variational_Damage_Gradient.ipynb)

### Simulation of regularized brittle fracture

- TDCB

### Simulation of regularized ductile fracture

- Explicit gradient model

- Implicit gradient model
