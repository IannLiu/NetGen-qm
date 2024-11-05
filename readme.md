## NetGen-qm: An [autodE](https://github.com/duartegroup/autodE) wrapper for automaticlly estimating kinetic parameters using transition state theory
This wrapper is developed for __automatically__ estimating kinetic parameters using transition state theory.  
The transition state search and thermodynamic search is based on [autodE](https://github.com/duartegroup/autodE). 
We developed additional codes for robustly searching the transition state of Hydrogen Abstraction and Bond Scission
reaction, calculating high-pressure-limit rate coefficient parameters, and fitting these parameters to the modified
Arrhenius formula.
#### To use the `NetGen-qm` code:
1. Installing autodE package:  
`conda install autode --channel conda-forge`
2. Installing the external electronic structure calculation package.   
   For ORCA software, "LD_LIBRARY_PATH" might not be inherited and the output file would be empty. To solve this problem,
   you should add the ORCA and openmpi lib to `os.environ`
   or [setting conda environment variables](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux)
3. Running the demo in `test/test.ipynb`  
Note: These codes provide a rough estimation of rate coefficient parameters. For more accurate estimations, users are 
recommended to use [ARC](https://github.com/ReactionMechanismGenerator/ARC).
