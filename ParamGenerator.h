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

class ParamGenerator {
    public:
    ParamGenerator(char*, char*, char*, char*);
    double getHx(int Z, int Ze);
    double getKxy(int Z1, int Ze1, int Z2, int Ze2);
    double getIP(int Z, int Ze);
    double getIP_BH(int Z, int Ze);
    double getEA(int Z, int Ze);
    double getEA_BH(int Z, int Ze);
    double getShell(int Z);
    
    private:
    char* ionization_potential_param;
    char* electron_affinity_param;
    char* gamma_param;
    char* beta_param;
};