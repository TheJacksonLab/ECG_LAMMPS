# PPP for LAMMPS (Beta)
Repository for implementation of a semi-empirical Fock solver for LAMMPS. It uses a combination of species based empirical parameters and interaction functions to generate the Fock matrix and obtain the molecular orbitals and energy levels. Most of the computationally intensive aspects are handled using GPU libraries for fast calculation. <br><br>
<strong><i> While the code is currently functional, it is still in active development. You may experience bugs or other unexpected output. Consult the authors if you are having trouble. </i></strong>

## How to compile the code
First, it is necessary to have a copy of LAMMPS, which can be obtained from the LAMMPS github repository. Then follow these instructions
1. In the src folder, create a folder called PPP.
2. Copy all the *.cpp and *.h from this repository into the PPP folder
3. Navigate to the cmake folder of lammps. Here there will be a file called CMakeLists.txt. Replace this with the CMakeLists.txt from this repository.
4. Build the lammps code using CMake, ensuring that you are asking for the PPP package to be build. Considering that you are building the code in a folder called lammps/build, This can be done using
```
   cmake -D PKG_PPP=yes ../cmake/
```
5. <strong> Note </strong>: This package is written for performance and is intended to be run on GPUs. It <i>requires</i> CUDA runtime, CuBLAS and Cusolver libraries.

## How to run the code.
To run PPP, you can specify the PPP fix in the input script. This can be done as
```
fix ID group ppp mode gamma_param beta_param ip_param ea_param maxiter norm_error etol units
run 1
unfix ID
```
Here
1. ``mode`` represents the type of Hartree-Fock. Set it as ``rhf`` for restricted HF. UHF has not yet been implemented
2. ``gamma_param`` represents the type of parametrization to be used for the electron repulsion interaction. Currently we only support the Beveridge-Hinze parametrization (J. Am. Chem. Soc. 1971, 93, 13, 3107–3114), which can be set using ``BH``.
3. ``beta_param`` represents the type of parametrization to be used for off-site interaction of the active electron with the core. Currently we only support Beveridge-Hinze (set using ``BH``)
4. ``ip_param`` and ``ea_param`` represent the data-set to be used for the ionzation potential and electron affinity. Currently we only support Beveridge-Hinze (set using ``BH``)
5. ``maxiter`` is the maximum number of SCF iterations
6. ``norm_error`` is the tolerance used to check for the convergence of the Pulay norm. This currently does not have any affect on the calculation, but in upcoming versions of the software it will be used during the DIIS calculation.
7. ``etol`` is the energy tolerance used to check for convergence. This is what we currently use to stop the SCF loop.
8. ``units`` represents what units are used for the coordinates, for example ''real'', ''metal'', ''cgs'', ''electron'', etc. Note that no matter the input units, the output energy levels will always be in eV.

Additional you also need a file called ``ppp_information.dat``, which lists the atomic number and number of active electrons for each atom type. For example, let's say you have two atom types 1 and 2. Atom type 1 represents conjugated carbon atoms which contribute 1 electron to the PPP calculation and atom type 2 represents Nitrogen which contributes 2 electrons to the PPP calculation. The ``ppp_information.dat`` file should look as follows
```
2
1 6 1
2 7 2
```

## Examples
The repository contains an examples folder which shows the LAMMPS PPP code being used on pentalene and azaphenalene.

## Contact
This package is being developed by Vishnu Raghuraman (vishnura@illinois.edu). Please feel free to reach out for user support, feedback or to learn how to contribute.
