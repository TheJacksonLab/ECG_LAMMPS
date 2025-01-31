/* -*- c++ -*- ----------------------------------------------------------
 * LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 * https://www.lammps.org/, Sandia National Laboratories
 * LAMMPS development team: developers@lammps.org
 *
 * Copyright (2003) Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 * certain rights in this software.  This software is distributed under
 * the GNU General Public License.
 *
 * See the README file in the top-level LAMMPS directory.
 * ------------------------------------------------------------------------- */


class GPUSolver{
    public:
    GPUSolver();
    double addtwonumbers(double, double);
    void solveFockMatrixCUDA(double *, int, double *);
    void buildDensityMatrixCUDA(double *, double *, double *, int, int, double, double);
    void constructBMatrix(double **, int , int , double *);
    void constructDIISFockMatrix(double *k, double **, double *,int, int);
    double calculateResidualMatrixCUDA(double *, double *, double *, int);
};
