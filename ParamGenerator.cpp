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

#include "ParamGenerator.h"
#include <cstring>

ParamGenerator::ParamGenerator(char* ipparam, char* eaparam, char* gparam, char* bparam)
{
    ionization_potential_param = ipparam;
    electron_affinity_param = eaparam;
    gamma_param = gparam;
    beta_param = bparam;
}

double ParamGenerator::getHx(int Z, int Ze)
{
    double h_x=0.0;
    if (Z == 6)
    {
        h_x= 0.00;
    }
    else if (Z == 7 && Ze == 1)
    {
        h_x = 0.51;
    }
    else if (Z == 7 && Ze == 2)
    {
        h_x = 1.37;
    }
    else if (Z == 8 && Ze == 1)
    {
        h_x = 0.97;
    }
    else if (Z == 8 && Ze == 2)
    {
        h_x = 2.09;
    }
    else if (Z == 9 && Ze == 2)
    {
        h_x = 2.71;
    }
    else if (Z == 15 && Ze == 1)
    {
        h_x = 0.19;
    }
    else if (Z == 15 && Ze == 2)
    {
        h_x = 0.75;
    }
    else if (Z == 16 && Ze == 1)
    {
        h_x = 0.46;
    }
    else if (Z == 16 && Ze == 2)
    {
        h_x = 1.11;
    }

    return h_x;
}

double ParamGenerator::getKxy(int Z1, int Ze1, int Z2, int Ze2)
{
    double k_xy=0.0;
    if (Z1 == 6)
    {
        if (Z2 == 6)
        {
            k_xy = 1;
        }
        if (Z2 == 7 && Ze2 == 1)
        {
            k_xy = 1.02;
        }
        if (Z2 == 7 && Ze2 == 2)
        {
            k_xy = 0.89;
        }
        if (Z2 == 16 && Ze2 == 1)
        {
            k_xy = 0.81;
        }
        if (Z2 == 16 && Ze2 == 2)
        {
            k_xy = 0.69;
        }
    }
    else if (Z1 == 7 && Ze1 == 1)
    {
        if (Z2 == 6)
        {
            k_xy = 1.02;
        }
        if (Z2 == 7 && Ze2 == 1)
        {
            k_xy = 1.09;
        }
        if (Z2 == 7 && Ze2 == 2)
        {
            k_xy = 0.99;
        }
    }
    else if (Z1 == 7 && Ze1 == 2)
    {
        if (Z2 == 6)
        {
            k_xy = 0.89;
        }
        if (Z2 == 7 && Ze2 == 1)
        {
            k_xy = 1.99;
        }
        if (Z2 == 7 && Ze2 == 2)
        {
            k_xy = 0.98;
        }
    }
    else if (Z1 == 16 && Ze1 == 1)
    {
        if (Z2 == 6)
        {
            k_xy = 0.81;
        }
        else if (Z2 == 16 && Ze2 == 1)
        {
            k_xy = 0.68;
        }
        else if (Z2 == 16 && Ze2 == 2)
        {
            k_xy = 0.58;
        }
    }
    else if (Z1 == 16 && Ze2 == 2)
    {
        if (Z2 == 6)
        {
            k_xy = 0.69;
        }
        else if (Z2 == 16 && Ze2 == 1)
        {
            k_xy = 0.58;
        }
        else if (Z2 == 16 && Ze2 == 2)
        {
            k_xy = 0.63;
        }
    }
    return k_xy;
}

double ParamGenerator::getIP(int Z, int Ze)
{
    double IP=0.0;
    if (strcmp(ionization_potential_param, "BH") == 0)
    {
        IP = getIP_BH(Z, Ze);
    }
    return IP;
}

double ParamGenerator::getIP_BH(int Z, int Ze)
{
    double IP=0.0;
    if (Z == 6)
    {
        IP = 11.16/27.2114079527;
    }
    else if (Z == 7 && Ze == 1)
    {
        IP = 14.12/27.2114079527;
    }
    else if (Z == 7 && Ze == 2)
    {
        IP = 28.71/27.2114079527;
    }
    else if (Z == 8 && Ze == 1)
    {
        IP = 17.70/27.2114079527;
    }
    else if (Z == 8 && Ze == 2)
    {
        IP = 34.12/27.2114079527;
    }
    else if (Z == 9 && Ze == 2)
    {
        IP = 40.07/27.2114079527;
    }
    else if (Z == 14 && Ze == 1)
    {
        IP = 9.17/27.2114079527;
    }
    else if (Z == 15 && Ze == 1)
    {
        IP = 11.64/27.2114079527;
    }
    else if (Z == 15 && Ze == 2)
    {
        IP = 20.68/27.2114079527;
    }
    else if (Z == 16 && Ze == 1)
    {
        IP = 12.70/27.2114079527;
    }
    else if (Z == 16 && Ze == 2)
    {
        IP = 23.74/27.2114079527;
    }
    return IP;
}

double ParamGenerator::getEA(int Z, int Ze)
{
    double EA=0.0;
    if (strcmp(electron_affinity_param, "BH") == 0)
    {
        EA = getEA_BH(Z, Ze);
    }
    return EA;
}

double ParamGenerator::getEA_BH(int Z, int Ze)
{
    double EA=0.0;
    if (Z == 6)
    {
        EA = 0.03/27.2114079527;
    }
    else if (Z == 7 && Ze == 1)
    {
        EA = 1.78/27.2114079527;
    }
    else if (Z == 7 && Ze == 2)
    {
        EA = 11.96/27.2114079527;
    }
    else if (Z == 8 && Ze == 1)
    {
        EA = 2.47/27.2114079527;
    }
    else if (Z == 8 && Ze == 2)
    {
        EA = 15.30/27.2114079527;
    }
    else if (Z == 9 && Ze == 2)
    {
        EA = 18.52/27.2114079527;
    }
    else if (Z == 14 && Ze == 1)
    {
        EA = 2.00/27.2114079527;
    }
    else if (Z == 15 && Ze == 1)
    {
        EA = 1.80/27.2114079527;
    }
    else if (Z == 15 && Ze == 2)
    {
        EA = 10.76/27.2114079527;
    }
    else if (Z == 16 && Ze == 1)
    {
        EA = 2.76/27.2114079527;
    }
    else if (Z == 16 && Ze == 2)
    {
        EA = 11.65/27.2114079527;
    }
    return EA;
}

double ParamGenerator::getShell(int Z)
{
    int n=0;
    if (Z < 10)
    {
        n = 2;
    }
    else if (Z < 20)
    {
        n = 3;
    }
    return n;
}