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


#ifdef FIX_CLASS
// clang-format off
FixStyle(ppp, FixPPP)
// clang-format on
#else

#ifndef LMP_FIX_PPP_H
#define LMP_FIX_PPP_H

#include "fix.h"
#include "GPUSolver.h"
#include "ParamGenerator.h"

namespace LAMMPS_NS {
class FixPPP : public Fix {
    public:
    FixPPP(class LAMMPS*, int ,char**);
    ~FixPPP();
    double identify_atom_type(double);
    int setmask() override;
    void end_of_step() override;
    double getIP_BH(int, int);
    double getIP_PS1(int, int);
    double getIP_PS2(int, int);
    double getIP(int, int, char*);
    double getEA_BH(int, int);
    double getEA(int, int, char*);
    void buildGammaMatrix();
    double calGammaSameSite(int, int);
    double calGammaCrossSiteBH(int, int);
    double getExponent(double);
    double factorial(int);
    double calA(int,double);
    double calB(int,double,double);
    double calSlaterOverlap(int atom1, int atom2);
    double calTwoPiTwoPiOverlap(double,double,double);
    double calTwoPiThreePiOverlap(double,double,double);
    double calThreePiThreePiOverlap(double,double,double);
    double calBetaBH(int,int);
    void buildBetaMatrix();
    double buildDensityMatrix();
    void buildCoreMatrix();
    double buildFockMatrix();
    double solveFockMatrix();
    void normalizeCoefficients();
    void calculateResidualVector(int);
    double calTotalEnergy();
    double getHx(int, int);
    double getKxy(int, int, int, int);
    void initializeCoefficients();
    void post_mortem();
    double **relevant_atoms;
    double **dist_matrix;
    double **connectivity_matrix;
    int **type_information;
    int num_types;
    int *num_local_atoms;
    int num_global_atoms;

    // PPP specifics
    char* mode; // RHF or UHF
    char* gamma_param; // What function to use for gamma
    char* beta_param; // What function to use for beta
    char* ip_param; // What dataset to use for ionization potential
    char* ea_param; // what dataset to use for electron affinity
    char *units;
    int maxiter; // Max number of SCF iteration
    double rms_tolerance; // Error at which SCF iteration stops
    double etolerance; // Energy tolerance
    double total_energy;
    int num_electrons;
    int num_mo;
    int ndiis;
    double nndist;
    double scf_error;
    double **coeff;
    double **density;
    double *coeff1dA, *coeff1dB, *density1d;
    double *fock1d;
    double **core;
    double **gamma;
    double **beta;
    double **fock;
    double **solution_vectors;
    double **residual_vectors;
    double *B;
    double *pulay_rhs;
    double *eigenvalues;
    class GPUSolver *solverObject;
    class ParamGenerator *paramObject;

    // temp variable
    double **temp_eigenvecs;
    double *prev_eigenvalues;
};
};
#endif
#endif