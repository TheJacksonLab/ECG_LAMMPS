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

#include "fix_ppp.h"
#include "atom.h"
#include "comm.h"
#include "text_file_reader.h"
#include "error.h"
#include "domain.h"
#include "math_eigen_impl.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathEigen;

typedef Jacobi<double, double *, double **, double const *const *> Jacobi_v2;
extern "C" {
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n,  const int* k,
               double*  alpha,  double *a, const int* lda, double *b, const int*
               ldb, double* beta, double *c, const int* ldc);
    void dsyev_( char* jobz, char* uplo, int* n, double *a, int* lda, double *w, double *work,
               int *lwork, int *info);
    void dsyevd_( char* jobz, char* uplo, int* n, double *a, int* lda, double *w, double *work,
               int *lwork, int *iwork, int *liwork, int *info);
    void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
}

FixPPP::FixPPP(class LAMMPS *lmp, int narg, char** arg) : Fix(lmp, narg, arg)
{
    // mode == 0 means RHF, mode == 1 means UHF
    mode = arg[3];

    // parameterization for gamma 
    gamma_param = arg[4];

    // parametrization for beta
    beta_param = arg[5];

    // ionization param
    ip_param = arg[6];

    // electron affinity param
    ea_param = arg[7];

    // max iter
    maxiter = utils::inumeric(FLERR, arg[8], false, lmp);

    // tolerance
    rms_tolerance = utils::numeric(FLERR, arg[9], false, lmp);

    // etolerance
    etolerance = utils::numeric(FLERR, arg[10], false, lmp);

    // distance units
    units = arg[11];

    // num of vectors to be used for DIIS
    //ndiis = utils::numeric(FLERR, arg[12], false, lmp);

    TextFileReader ppp_information("ppp_information.dat", "ppp_information");
    char *ntypes = ppp_information.next_line(1);
    ValueTokenizer vt(ntypes, "\n");
    num_types = vt.next_int();

    type_information = new int*[num_types];
    for (int i=0;i<num_types;i++)
    {
        type_information[i] = new int[2];
        char *type_info = ppp_information.next_line(3);
        ValueTokenizer vt(type_info);
        int type = vt.next_int();
        type_information[i][0] = vt.next_int();
        type_information[i][1] = vt.next_int();
    }

    // Now we have to do the somewhat tedious work of collecting all the relevant atoms
    // i.e pi-bonded carbons, nitrogens etc across the different MPI ranks

    // First we ascertain how many revelant atoms are present on each MPI rank
    // The idea is to create an array initialized to zero, and each MPI process will fill one of the indices
    // Then we use an MPI_Allreduce command to distribute that information across ranks and fill the array completely.

    num_local_atoms = new int[comm->nprocs];
    for (int i=0;i<comm->nprocs;i++)
    {
        num_local_atoms[i] = 0;
    }
    //memset(num_local_atoms, 0, comm->nprocs*sizeof(int));
    num_global_atoms = 0;

    // vector that will store all the global indices for each atom on its MPI rank
    std::vector<int> global_id_vector = {};
    for (int j=0;j<atom->nlocal;j++)
    {
        int atom_type = atom->type[j];
        if (type_information[atom->type[j]-1][1] != 0)
        {
            num_local_atoms[comm->me] = num_local_atoms[comm->me]+1;
            global_id_vector.push_back(atom->tag[j]);
        }
    }

    // Get all the information across ranks
    MPI_Allreduce(MPI_IN_PLACE, num_local_atoms, comm->nprocs, MPI_INT, MPI_SUM, world);

    // Ascertain what is the max number of atoms on a single processer
    // This is important so that we know how much memory we need to allocate
    int max_local_atoms = 0.0;
    for (int i=0;i<comm->nprocs;i++)
    {
        max_local_atoms = std::max(max_local_atoms, num_local_atoms[i]);
        num_global_atoms = num_global_atoms + num_local_atoms[i];
    }

    // Here we will store the global indices of the relevant atoms across each MPI rank
    // The idea is the same as before, create a big array, each MPI rank will fill parts of it
    // Then we collect all that using MPI_Allreduce


    int tags_of_local_atoms[comm->nprocs][max_local_atoms];
    memset(tags_of_local_atoms, 0, comm->nprocs*max_local_atoms*sizeof(int));
    for (int j=0;j<num_local_atoms[comm->me];j++)
    {
        tags_of_local_atoms[comm->me][j] = global_id_vector[j];
    }

    MPI_Allreduce(MPI_IN_PLACE, tags_of_local_atoms, comm->nprocs*max_local_atoms, MPI_INT, MPI_SUM, world);


    // Ok finally we have the identities of all the relevant atoms present across all processors
    // Our job is done, nearly
    // We now need to access these atoms' position,atomic number, etc
    // And store it in a nicer data structure

    // Initialize and ensure everything is zero
    relevant_atoms = new double* [num_global_atoms];
    dist_matrix = new double* [num_global_atoms];
    connectivity_matrix = new double* [num_global_atoms];
    for (int i=0;i<num_global_atoms;i++)
    {
        relevant_atoms[i] = new double [5];
        for (int j=0;j<5;j++)
        {
            relevant_atoms[i][j] = 0.0;
        }
        dist_matrix[i] = new double [num_global_atoms];
        connectivity_matrix[i] = new double [num_global_atoms];
    }
    // Note that we want the positions in atomic units, but that may not be the input given
    // We need to calculate a pre-factor to ensure that the final coordinates are in atomic units

    double pre_factor=0.0;
    if (strcmp(units,"real") == 0)
    {
        pre_factor = 1.0/0.529177210903;
    }
    else if (strcmp(units,"metal") == 0)
    {
        pre_factor = 1.0/0.529177210903;
    }
    else if (strcmp(units,"si") == 0)
    {
        pre_factor = 1e10 * 1.0/0.529177210903;
    }
    else if (strcmp(units,"cgs") == 0)
    {
        pre_factor = 1e8 * 1.0/0.52917710903;
    }
    else if (strcmp(units,"electron") == 0)
    {
        pre_factor = 1.0;
    }
    else if (strcmp(units,"micro") == 0)
    {
        pre_factor = 1e4 * 1.0/0.5291770903;
    }
    else if (strcmp(units,"nano") == 0)
    {
        pre_factor = 10/0.5291770903;
    }
    else
    {
        error->all(FLERR, "Unrecognized units, please pick one of the standard options given in LAMMPS documentation");
    }


    // Each rank fills its part of the array
    // CORRECTION:- The better way of doing it gives a segmentation fault,so we're doing it in a more stupid way
    // Everytime we find an atom on a particular processor, the information of that atom is sent across all processors
    // This happens in the atomdata array
    // Finally this atomdata array is copied to the relevant_atoms main structure
    int gindex=0;
    num_electrons=0;
    for (int i=0;i<comm->nprocs;i++)
    {
        for (int j=0;j<num_local_atoms[i];j++)
        {
            int global_index = tags_of_local_atoms[i][j];
            double atomdata[5];
            memset(atomdata, 0.0, 5*sizeof(double));
            for (int k=0;k<atom->nlocal;k++)
            {
                if (atom->tag[k] == global_index)
                {
                    atomdata[0] = pre_factor*atom->x[k][0];
                    atomdata[1] = pre_factor*atom->x[k][1];
                    atomdata[2] = pre_factor*atom->x[k][2];
                    atomdata[3] = type_information[atom->type[k]-1][0];
                    atomdata[4] = type_information[atom->type[k]-1][1];
                    break;
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, atomdata, 5, MPI_DOUBLE, MPI_SUM, world);
            relevant_atoms[gindex][0] = atomdata[0];
            relevant_atoms[gindex][1] = atomdata[1];
            relevant_atoms[gindex][2] = atomdata[2];
            relevant_atoms[gindex][3] = atomdata[3];
            relevant_atoms[gindex][4] = atomdata[4];
            num_electrons =num_electrons + atomdata[4];
            gindex=gindex+1;
        }
    }

    //print
    /*
    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<5;j++)
        {
            fmt::print(screen, "{} ", relevant_atoms[i][j]);
        }
        fmt::print(screen, "{}\n", " ");
    }
    */

    // Let's calculate distance matrix
    nndist = 1e5 * 1.1;
    double lattice_x = pre_factor*(domain->boxhi[0] - domain->boxlo[0]);
    double lattice_y = pre_factor*(domain->boxhi[1] - domain->boxlo[1]);
    double lattice_z = pre_factor*(domain->boxhi[2] - domain->boxlo[2]);
    for (int i=0;i<num_global_atoms;i++)
    {
        dist_matrix[i][i] = 0.0;
        for (int j=i+1;j<num_global_atoms;j++)
        {
            double min_x = std::min(fabs(relevant_atoms[i][0] - relevant_atoms[j][0]),
            std::min(fabs(relevant_atoms[i][0] - relevant_atoms[j][0] + lattice_x), fabs(relevant_atoms[i][0] - relevant_atoms[j][0] - lattice_x)));

            double min_y = std::min(fabs(relevant_atoms[i][1] - relevant_atoms[j][1]),
            std::min(fabs(relevant_atoms[i][1] - relevant_atoms[j][1] + lattice_y), fabs(relevant_atoms[i][1] - relevant_atoms[j][1] - lattice_y)));

            double min_z = std::min(fabs(relevant_atoms[i][2] - relevant_atoms[j][2]),
            std::min(fabs(relevant_atoms[i][2] - relevant_atoms[j][2] + lattice_z), fabs(relevant_atoms[i][2] - relevant_atoms[j][2] - lattice_z)));

            dist_matrix[i][j] = sqrt(pow(min_x, 2) + pow(min_y, 2) + pow(min_z, 2));
            dist_matrix[j][i] = dist_matrix[i][j];
            if (dist_matrix[i][j] < pre_factor*2)
            {
                connectivity_matrix[i][j] = 1.0;
                connectivity_matrix[j][i] = 1.0;
            }
            else 
            {
                connectivity_matrix[i][j] = 0.0;
                connectivity_matrix[j][i] = 0.0;
            }
            nndist = std::min(dist_matrix[i][j], nndist);
        }
    }

    //dist matrix
    /*
    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            fmt::print(screen, "{} ", dist_matrix[i][j]);
        }
        fmt::print(screen, "{}\n", " ");
    }
    */

    //int bufsize = num_global_atoms * 5;
    // A final MPI_Allreduce to bring it all home
    // MPI_Allreduce( MPI_IN_PLACE, relevant_atoms, 2, MPI_DOUBLE, MPI_MAX, world );
    //
    // THIS SHOULD HAVE WORKED, BUT INSTEAD GIVES A SEGFAULT. MEMORY OVERFLOW PROBLEM? MOST LIKELY
    //

    // Ok we finally have the atoms, let's start the procedure
    // Identifying number of orbitals
    if (strcmp(mode, "RHF") == 0 || strcmp(mode, "rhf") == 0)
    {
        if (num_electrons % 2 != 0)
        {
            error->all(FLERR, "For RHF, even number of electrons are needed! Try again");
        }
        num_mo = num_electrons/2;
    }
    else if (strcmp(mode, "UHF") == 0 || strcmp(mode, "uhf") == 0)
    {
        // add some uhf stuff here later
    }
    else
    {
        error->all(FLERR, "Unknown mode: pick RHF or UHF");
    }

    // Allocate matrices for PPP calculation
    coeff = new double* [num_mo];
    for (int i=0;i<num_mo;i++)
    {
        coeff[i] = new double [num_global_atoms];
    }

    beta = new double* [num_global_atoms];
    gamma = new double* [num_global_atoms];
    core = new double* [num_global_atoms];
    density = new double* [num_global_atoms];
    fock = new double* [num_global_atoms];
    eigenvalues = new double [num_global_atoms];

    for (int i=0;i<num_global_atoms;i++)
    {
        beta[i] = new double [num_global_atoms];
        gamma[i] = new double [num_global_atoms];
        core[i] = new double [num_global_atoms];
        density[i] = new double [num_global_atoms];
        fock[i] = new double [num_global_atoms];
    }

    // allocate memory for DIIS (solution, residual, B)
    /*
    solution_vectors = new double*[ndiis];
    residual_vectors = new double*[ndiis];
    for (int i=0;i<ndiis;i++)
    {
        solution_vectors[i] = new double [num_global_atoms*num_global_atoms];
        residual_vectors[i] = new double [num_global_atoms*num_global_atoms];
    }
    B = new double [(ndiis+1)*(ndiis+1)];
    pulay_rhs = new double [ndiis+1];
    for (int i=0;i<ndiis;i++)
    {
        pulay_rhs[i] = 0.0;
    }
    pulay_rhs[ndiis] = -1.0;
    */

    // some necessary checks because users are idiots
    if (strcmp(gamma_param, "BH") != 0  && strcmp(gamma_param, "Ohno") != 0 && strcmp(gamma_param, "MN") != 0)
    {
        error->all(FLERR, "The gamma parametrization choice doesn't match available options! Pick either BH, Ohno or MN");
    }

    if (strcmp(beta_param, "BH") != 0 && strcmp(beta_param, "MN") != 0)
    {
        error->all(FLERR, "The beta parametrization choice doesn't match available options! Pick either BH or MN");
    }

    if (strcmp(ip_param, "BH") != 0 && strcmp(ip_param, "PS1") != 0 && strcmp(ip_param, "PS2") != 0)
    {
        error->all(FLERR, "The IP choice doesn't match available options! Pick either BH, PS1 or PS2");
    }

    if (strcmp(ea_param, "BH") != 0 && strcmp(ea_param, "PS1") != 0 && strcmp(ea_param, "PS2") != 0)
    {
        error->all(FLERR, "The EA choice doesn't match available options! Pick either BH, PS1, or PS2");
    }


    if (comm->me == 0)
    {   
      fmt::print(screen, "{}\n", "###############################################");
      fmt::print(screen, "{}\n", "               PPP INITIALIZATION              ");
      fmt::print(screen, "{}\n", "###############################################");
      fmt::print(screen,"{} {}\n","Number of atoms: ", num_global_atoms);
      fmt::print(screen,"{} {}\n","Number of electrons: ", num_electrons);
      fmt::print(screen,"{} {}\n", "Number of orbitals: ", num_mo);
      fmt::print(screen,"{} {}\n","Mode: ", mode);
      fmt::print(screen,"{} {}\n","Gamma parametrization: ", gamma_param);
      fmt::print(screen,"{} {}\n","Beta parametrization: ", beta_param);
      fmt::print(screen,"{} {}\n","Ionization Potential choice: ",ip_param);
      fmt::print(screen,"{} {}\n","Electron Affinity choice: ", ea_param);
      fmt::print(screen,"{} {}\n","Max number of iterations: ",maxiter);
      fmt::print(screen,"{} {}\n","Error tolerance: ",rms_tolerance);
      fmt::print(screen,"{} {}\n","Energy tolerance: ",etolerance);
      fmt::print(screen, "{}\n", "###############################################");
    }

    temp_eigenvecs = new double *[num_global_atoms];
    prev_eigenvalues = new double [num_global_atoms];
    for (int i=0;i<num_global_atoms;i++)
    {
        temp_eigenvecs[i] = new double [num_global_atoms];
        prev_eigenvalues[i] = 0.0;
    }

    // do Huckel calculation as starting point
    solverObject = new GPUSolver;
    paramObject = new ParamGenerator(ip_param, ea_param, gamma_param, beta_param);

    initializeCoefficients();

    scf_error = 0.0;
    coeff1dA = new double [num_mo*num_global_atoms];
    coeff1dB = new double [num_mo*num_global_atoms];
    density1d = new double [num_global_atoms*num_global_atoms];
    fock1d = new double [num_global_atoms*num_global_atoms];

    
}

void FixPPP::initializeCoefficients()
{
    double *huckel, *huckel_eigenvals;
    huckel_eigenvals = new double[num_global_atoms];
    huckel = new double[num_global_atoms*num_global_atoms];
    //for (int i=0;i<num_global_atoms;i++)
   // {
    //    huckel[i] = new double [num_global_atoms];
   // }
    for (int i=0;i<num_global_atoms;i++)
    {
        huckel[i*num_global_atoms+i] = paramObject->getHx(relevant_atoms[i][3], relevant_atoms[i][4]);
        for (int j=i+1;j<num_global_atoms;j++)
        {
           if (connectivity_matrix[i][j] == 1) 
           {
                huckel[i*num_global_atoms + j] = paramObject->getKxy(relevant_atoms[i][3], relevant_atoms[i][4], relevant_atoms[j][3], relevant_atoms[j][4]);
                huckel[j*num_global_atoms + i] = huckel[i*num_global_atoms + j];
            }
           else
           {
                huckel[i*num_global_atoms + j] = 0.0;
                huckel[j*num_global_atoms + i] = 0.0;
           }
        }

    }

    /*
    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            fmt::print(screen, "{} ", huckel[i*num_global_atoms + j]);
        }
        fmt::print(screen, "{}", "\n");
    }

    fmt::print(screen, "Calculated the huckel matrix\n");
    */

    char jobz='V';
    char uplo='U';
    double *work;
    int *iwork;
    int info;
    int lwork=-1;
    int liwork=-1;
    double temp_work;
    int temp_iwork;

    
    //workspace query
    /*
    dsyevd_(&jobz, &uplo, &num_global_atoms, huckel, &num_global_atoms, huckel_eigenvals, &temp_work,
            &lwork, &temp_iwork, &liwork, &info);
    lwork = temp_work;
    liwork = temp_iwork;
    work = new double [lwork];
    iwork = new int [liwork];
    dsyevd_( &jobz, &uplo, &num_global_atoms, huckel, &num_global_atoms, huckel_eigenvals,  work,
            &lwork, iwork, &liwork, &info);
    */
    solverObject->solveFockMatrixCUDA(huckel, num_global_atoms, huckel_eigenvals);
    //Jacobi_v2 hc_solver(num_global_atoms);
    //hc_solver.Diagonalize(huckel, huckel_eigenvals, temp_eigenvecs, Jacobi_v2::SORT_DECREASING_EVALS, true, 1000);

    //for (int i=num_global_atoms-1;i>num_global_atoms-1-num_mo;i--)
    //{
    //    fmt::print(screen, "{}\n", huckel_eigenvals[i]);
   // }

    for (int i=0;i<num_mo;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            coeff[i][j] = huckel[(num_global_atoms-1-i)*num_global_atoms + j];//temp_eigenvecs[i][j];
        }
    }

}

int FixPPP::setmask()
{
    int mask=0;
    mask |= FixConst::END_OF_STEP;
    return mask;
}

double FixPPP::calGammaSameSite(int Z, int Ze)
{
    double IP = paramObject->getIP(Z, Ze);
    double EA = paramObject->getEA(Z, Ze);
    return IP-EA;
}

double FixPPP::calGammaCrossSiteBH(int atom1, int atom2)
{
    double gamma_11 = calGammaSameSite(relevant_atoms[atom1][3], relevant_atoms[atom1][4]);
    double gamma_22 = calGammaSameSite(relevant_atoms[atom2][3], relevant_atoms[atom2][4]);
    double a12 = 2.0/(gamma_11 + gamma_22);
    double r12 = dist_matrix[atom1][atom2];
    double gamma_12 = 1.0/(a12*exp(-0.5*pow(r12,2)/pow(a12,2)) + r12);
    return gamma_12;
}

void FixPPP::buildGammaMatrix()
{
    for (int i=0;i<num_global_atoms;i++)
    {
        gamma[i][i] = calGammaSameSite(relevant_atoms[i][3],relevant_atoms[i][4]);
        if (strcmp(gamma_param, "BH") == 0)
        {
            for (int j=i+1;j<num_global_atoms;j++)
            {
                gamma[i][j] = calGammaCrossSiteBH(i, j);
                gamma[j][i] = gamma[i][j];
            }
        }
    }
}

// slater exponent for overlap element
double FixPPP::getExponent(double gamma)
{
    double exponent = (1280.0/501.0)*gamma;
    return exponent;
}

double FixPPP::factorial(int n)
{
    if (n == 1 || n == 0)
    {
        return 1.0;
    }
    else
    {
        return n*factorial(n-1);
    }
}

double FixPPP::calA(int k, double p)
{
    double diff_sum = 0.0;
    for (int i=1;i<=k+1;i++)
    {
        diff_sum = diff_sum + factorial(k)/(pow(p, i) * factorial(k - i + 1));
    }
    return exp(-1.0*p)*diff_sum;
}

double FixPPP::calB(int k, double p, double t)
{
    double diff_sum = 0.0;
    for (int i=1;i<=k+1;i++)
    {
        diff_sum = diff_sum + (1 + pow(-1.0, k - i) * exp(2.0*p*t))*(factorial(k)/(pow(p*t, i) * factorial(k - i + 1)));
    }
    return (-exp(-1.0*p*t))*diff_sum;
}


double FixPPP::calSlaterOverlap(int atom1, int atom2)
{
    double overlap=0.0;
    double gamma_1 = gamma[atom1][atom1];
    double gamma_2 = gamma[atom2][atom2];
    double dist_12 = dist_matrix[atom1][atom2];
    int shell1 = paramObject->getShell(relevant_atoms[atom1][3]);
    int shell2 = paramObject->getShell(relevant_atoms[atom2][3]);
    if (connectivity_matrix[atom1][atom2] == 0)
    {
        return 0.0;
    }
    else
    {
    if (shell1 == 2 && shell2 == 2)
    {
        overlap = calTwoPiTwoPiOverlap(gamma_1, gamma_2, dist_12);
    }
    else if (shell1 == 3 && shell2 == 3)
    {
        overlap = calThreePiThreePiOverlap(gamma_1, gamma_2, dist_12);
    }
    else
    {
        overlap = calTwoPiThreePiOverlap(gamma_1, gamma_2, dist_12);
    }
    return overlap;
    }
}

double FixPPP::calTwoPiTwoPiOverlap(double gamma_1,double gamma_2,double dist_12)
{
    double overlap=0.0;
    

    double exponent1 = getExponent(gamma_1);
    double exponent2 = getExponent(gamma_2);
    double p = 0.5*dist_12*(exponent1 + exponent2);
    double t = (exponent1 - exponent2)/(exponent1 + exponent2);
    if (p == 0)
    {
        overlap = pow((1 - pow(t, 2)), 2.5);
    }
    else if (t == 0)
    {
        overlap = exp(-1.0*p)*(1.0 + p  + 0.4*pow(p, 2) + (1.0/15.0)*pow(p, 3));
    }
    else
    {
        overlap = (1.0/32.0) * pow(p, 5) * pow((1.0 - pow(t,2)), 2.5) * (calA(4, p) * (calB(0,p,t) - calB(2,p,t)) + 
        calA(2,p)*(calB(4,p,t) - calB(0,p,t)) + calA(0,p)*(calB(2,p,t) - calB(4,p,t)));
    }
    return overlap;
}

double FixPPP::calTwoPiThreePiOverlap(double gamma_1, double gamma_2, double dist_12)
{
    double overlap=0.0;

    
    double exponent1 = getExponent(gamma_1);
    double exponent2 = getExponent(gamma_2);
    double p = 0.5*dist_12*(exponent1 + exponent2);
    double t = (exponent1 - exponent2)/(exponent1 + exponent2);
    // p = 0 or t = 0 is not possible here, so just move on to general case
    overlap = (1.0/(32.0 * sqrt(30.0)))* pow(p, 6)*pow(1 + t, 2.5)*pow(1-t,3.5)*
    (calA(5, p)*(calB(0,p,t) - calB(2,p,t)) + calA(4, p)*(calB(3,p,t) - calB(1,p,t)) + 
    calA(3,p)*(calB(4,p,t) - calB(0,p,t)) + calA(2,p)*(calB(1,p,t) - calB(5,p,t)) +
    calA(1,p)*(calB(2,p,t) - calB(4,p,t)) + calA(0,p)*(calB(5,p,t) - calB(3,p,t)));
    return overlap;
}

double FixPPP::calThreePiThreePiOverlap(double gamma_1, double gamma_2, double dist_12)
{
    double overlap=0.0;
    
    double exponent1 = getExponent(gamma_1);
    double exponent2 = getExponent(gamma_2);
    double p = 0.5*dist_12*(exponent1 + exponent2);
    double t = (exponent1 - exponent2)/(exponent1 + exponent2);
    if (t == 0)
    {
        overlap = exp(-1.0*p)*(1 + p + (34.0/75.0)*pow(p,2) + (3.0/25.0)*pow(p,3) 
        + (31.0/1575.0)*pow(p,4) + (1.0/525.0)*pow(p,5));
    }
    else if (p == 0)
    {
        overlap = pow(1.0 - pow(t, 2), 3.5);
    }
    else
    {
        overlap = (1.0/960.0)*pow(p, 7)*pow(1 - pow(t,2), 3.5)*
        (calA(6, p)*(calB(0,p,t) - calB(2,p,t)) + calA(4,p)*(2*calB(4,p,t) - calB(0,p,t) - calB(2,p,t))
        + calA(2,p)*(2*calB(2,p,t) - calB(4,p,t) - calB(6,p,t)) + calA(0,p)*(calB(6,p,t) - calB(4,p,t)));
    }
    return overlap;
}

double FixPPP::calBetaBH(int atom1, int atom2)
{
    double gamma_12 = gamma[atom1][atom2];
    double overlap = calSlaterOverlap(atom1, atom2);
    double beta = 0.5*(relevant_atoms[atom1][4] + relevant_atoms[atom2][4])*overlap*(gamma_12 - (1.09/dist_matrix[atom1][atom2]));
    return beta;
}

void FixPPP::buildBetaMatrix()
{
    for (int i=0;i<num_global_atoms;i++)
    {
        beta[i][i] = 0.0;
        if (strcmp(beta_param, "BH") == 0)
        {
            for (int j=i+1;j<num_global_atoms;j++)
            {
                beta[i][j] = calBetaBH(i, j);
                beta[j][i] = beta[i][j];
            }
        }
    }
}

double FixPPP::buildDensityMatrix()
{
    double start = platform::walltime();

    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<num_mo;j++)
        {
            coeff1dA[i*num_mo + j] = coeff[j][i];
            coeff1dB[i*num_mo + j] = coeff[j][i];
        }
    }

    /*
    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            density[i][j] = 0.0;
            for (int k=0;k<num_mo;k++)
            {
                density[i][j] = density[i][j] + 2.0 * coeff[k][i] * coeff[k][j];
            }
            //fmt::print(screen, "{} ", density[i][j]);
        }
        //fmt::print(screen, "{}", "\n");
    }
    */
   char transA = 'T';
   char transB = 'N';
   double alpha = 2.0;
   double beta = 0.0;
   solverObject->buildDensityMatrixCUDA(coeff1dA, coeff1dB, density1d, num_global_atoms, num_mo, alpha, beta);
   //dgemm_(&transA, &transB, &num_global_atoms, &num_global_atoms, &num_mo, &alpha, coeff1dA, 
   //&num_mo, coeff1dB, &num_mo, &beta, density1d, &num_global_atoms);

   
   for (int i=0;i<num_global_atoms;i++)
   {
    for (int j=i;j<num_global_atoms;j++)
    {
        density[i][j] = density1d[i*num_global_atoms + j];
        density[j][i] = density[i][j];
    }
   }
   

    double end = platform::walltime();
    return end-start;
}

void FixPPP::buildCoreMatrix()
{
    double ip_i=0.0;
    for (int i=0;i<num_global_atoms;i++)
    {
        ip_i = paramObject->getIP(relevant_atoms[i][3], relevant_atoms[i][4]);
        core[i][i] = -ip_i;
        for (int j=0;j<num_global_atoms;j++)
        {
            if (j != i)
            {
                core[i][i] = core[i][i] - relevant_atoms[j][4]*gamma[i][j];
                core[i][j] = beta[i][j];
            }
            
        }
    }
}

double FixPPP::buildFockMatrix()
{
    double start = platform::walltime();
    #pragma omp parallel for
    for (int i=0;i<num_global_atoms;i++)
    {
        fock[i][i] = core[i][i] + 0.5*density[i][i]*gamma[i][i];
        for (int j=0;j<num_global_atoms;j++)
        {
            if (i!=j)
            {
                fock[i][i] = fock[i][i] + density[j][j]*gamma[i][j];
                fock[i][j] = core[i][j] - 0.5*density[i][j]*gamma[i][j];
            }
        }
    }
    double end = platform::walltime();
    return end-start;
}


double FixPPP::solveFockMatrix()
{
    double start = platform::walltime();

    char jobz='V';
    char uplo='U';
    double *work;
    int *iwork;
    int info;
    int lwork=-1;
    int liwork=-1;
    double temp_work;
    int temp_iwork;

    solverObject->solveFockMatrixCUDA(fock1d, num_global_atoms, eigenvalues);
    /*
    Jacobi_v2 solver(num_global_atoms);
    solver.Diagonalize(fock, eigenvalues, temp_eigenvecs, Jacobi_v2::SORT_INCREASING_EVALS, true, 1000);

    //workspace query
    dsyevd_(&jobz, &uplo, &num_global_atoms, fock1d, &num_global_atoms, eigenvalues, &temp_work,
            &lwork, &temp_iwork, &liwork,&info);
    lwork = temp_work;
    liwork = temp_iwork;
    work = new double [lwork];
    iwork = new int [liwork];
    double start = platform::walltime();
    dsyevd_( &jobz, &uplo, &num_global_atoms, fock1d, &num_global_atoms, eigenvalues,  work,
            &lwork, iwork, &liwork, &info);
    double end = platform::walltime();
    */
   
    scf_error = 0.0;
    
    for (int i=0;i<num_global_atoms;i++)
    {
        scf_error = scf_error + fabs(eigenvalues[i] - prev_eigenvalues[i]);
        prev_eigenvalues[i] = eigenvalues[i];
    }

    
    for (int i=0;i<num_mo;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            coeff[i][j] = fock1d[i*num_global_atoms + j];//temp_eigenvecs[i][j];
        }
    }

    double end = platform::walltime();
    return end-start;

}

void FixPPP::normalizeCoefficients()
{
    double sum, norm;
    for (int i=0;i<num_mo;i++)
    {
        sum=0.0;
        for (int j=0;j<num_global_atoms;j++)
        {
            sum = sum + pow(coeff[i][j], 2);
        }
        norm = sqrt(sum);
        for (int j=0;j<num_global_atoms;j++)
        {
            coeff[i][j] = (1.0/norm)*coeff[i][j];
        }
    }
}

double FixPPP::calTotalEnergy()
{
    double energy=0.0;
    #pragma omp parallel for collapse(2)
    for (int i=0;i<num_global_atoms;i++)
    {
        for (int j=0;j<num_global_atoms;j++)
        {
            energy = energy + 0.5*density[i][j]*(core[i][j] + fock[i][j]);
        }
    }
    return energy;
}


void FixPPP::end_of_step()
{
    double buildDensityTime, buildFockTime, solveFockTime;
    buildGammaMatrix();
    buildBetaMatrix();
    buildCoreMatrix();

    // DIIS helper variables
    int looped_iter;
    //int bmatsize=ndiis+1;
    //int nrhs=1;
    //int lda =bmatsize;
    //int ldb =bmatsize;
    //int info;
    //int *ipiv;
    //ipiv = new int[bmatsize];

    //tolerance helper variables
    double ediff=0.0;
    double norm=0.0;
    double new_energy=0.0;
    total_energy=0.0;

    if (comm->me == 0)
    {
        fmt::print(screen, "\n\n{}\n\n", "Warm-Up Phase");
    }

    //iterations, warm-up
    for (int iter=-5;iter<0;iter++)
    {
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Iteration: ", iter+1);
        }
        buildDensityTime = buildDensityMatrix();
        buildFockTime = buildFockMatrix();

        //make 1D fock for solving use and add it to DIIS structure
        #pragma omp parallel for collapse(2)
        for (int i=0;i<num_global_atoms;i++)
        {
            for (int j=0;j<num_global_atoms;j++)
            {
                fock1d[i*num_global_atoms + j] = fock[j][i];
            }
        }

        solveFockTime = solveFockMatrix();
        
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Total energy: ", total_energy);
            fmt::print(screen, "{} {}\n", "buildDensityTime: ", buildDensityTime);
            fmt::print(screen, "{} {}\n", "buildFockTime:", buildFockTime);
            fmt::print(screen, "{} {}\n", "solveFockTime:", solveFockTime);
        }
        normalizeCoefficients();
    }

    //  iterations, pre-diis
    if (comm->me == 0)
    {
        fmt::print(screen, "\n\n{}\n\n", "Pre-DIIS Phase");
    }


    for (int iter=0;iter<maxiter;iter++)
    {
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Iteration: ", iter+1);
        }
        buildDensityTime = buildDensityMatrix();
        buildFockTime = buildFockMatrix();

        //make 1D fock for solving use and add it to DIIS structure
        #pragma omp parallel for collapse(2)
        for (int i=0;i<num_global_atoms;i++)
        {
            for (int j=0;j<num_global_atoms;j++)
            {
                fock1d[i*num_global_atoms + j] = fock[j][i];
                //solution_vectors[iter][i*num_global_atoms+j] = fock[j][i];
            }
        }

        //norm = solverObject->calculateResidualMatrixCUDA(density1d, fock1d, residual_vectors[iter], num_global_atoms);
        new_energy = calTotalEnergy();
        ediff = fabs(new_energy-total_energy);
        total_energy=new_energy;
        solveFockTime = solveFockMatrix();
        
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Total energy: ", total_energy);
            fmt::print(screen, "{} {}\n", "Ediff: ", ediff);
            fmt::print(screen, "{} {}\n", "Norm error:",norm);
            fmt::print(screen, "{} {}\n", "buildDensityTime: ", buildDensityTime);
            fmt::print(screen, "{} {}\n", "buildFockTime:", buildFockTime);
            fmt::print(screen, "{} {}\n", "solveFockTime:", solveFockTime);
        }
        if (ediff < etolerance)
        {
            break;
        }
        normalizeCoefficients();
    }
    if (comm->me == 0)
    {
        fmt::print(screen, "\n\n{}\n\n", "DIIS Phase");
    }
    // Start DIIS
    /*
    for (int iter=ndiis-1;iter<maxiter;iter++)
    {
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Iteration: ", iter+1);
        }      
        buildDensityTime = buildDensityMatrix();
        buildFockTime = buildFockMatrix(); 
        looped_iter = iter % ndiis;

        //make 1D fock for solving use and add it to DIIS structure
        #pragma omp parallel for collapse(2)
        for (int i=0;i<num_global_atoms;i++)
        {
            for (int j=0;j<num_global_atoms;j++)
            {
                
                fock1d[i*num_global_atoms + j] = fock[j][i];
                solution_vectors[looped_iter][i*num_global_atoms+j] = fock[j][i];
            }
        }

        // add residual matrix
        norm = solverObject->calculateResidualMatrixCUDA(density1d, fock1d, residual_vectors[looped_iter], 
        num_global_atoms);
        new_energy = calTotalEnergy();
        ediff = fabs(new_energy-total_energy);
        total_energy=new_energy;

        // construct B matrix
        double start = platform::walltime();
        solverObject->constructBMatrix(residual_vectors, num_global_atoms, ndiis, B);
        
        // reset the pulay_rhs vector
        for (int i=0;i<ndiis;i++)
        {
            pulay_rhs[i] = 0.0;
        }
        pulay_rhs[ndiis] = -1.0;

        // solve the linear system to get DIIS coefficients
        dgesv_(&bmatsize, &nrhs, B, &lda, ipiv, pulay_rhs, &ldb, &info);

        // construct new DIIS fock matrix
        
        solverObject->constructDIISFockMatrix(fock1d, solution_vectors, pulay_rhs, ndiis, num_global_atoms);
        double end = platform::walltime();

        solveFockTime = solveFockMatrix();
        
        if (comm->me == 0)
        {
            fmt::print(screen, "{} {}\n", "Total energy: ", total_energy);
            fmt::print(screen, "{} {}\n", "Ediff: ", ediff);
            fmt::print(screen, "{} {}\n", "Norm error:",norm);
            fmt::print(screen, "{} {}\n", "buildDensityTime: ", buildDensityTime);
            fmt::print(screen, "{} {}\n", "buildFockTime:", buildFockTime);
            fmt::print(screen, "{} {}\n", "solveFockTime:", solveFockTime);
            fmt::print(screen, "{} {}\n", "DIISTime:", end-start);
        }
        if (norm < rms_tolerance && ediff < etolerance)
        {
            break;
        }
        normalizeCoefficients();
    }
    */
}

void FixPPP::post_mortem()
{
    if (comm->me == 0)
    {
        fmt::print(screen, "{}\n", "###############################################");
        fmt::print(screen, "{}\n", "              PPP OUTPUT SUMMARY               ");
        fmt::print(screen, "{}\n", "###############################################");
        FILE *ftype = fopen("eigenvalues.dat", "w");
        fprintf(ftype, "Occupied\n");
        fmt::print(screen, "{}\n", "Occupied");
        for (int i=0;i<num_mo;i++)
        {
            fprintf(ftype, "%f\n", eigenvalues[i]*27.2);
            fmt::print(screen, "{}\n", eigenvalues[i]*27.2);
        }
        fprintf(ftype, "Unoccupied\n");
        fmt::print(screen, "{}\n", "Unoccupied");
        for (int i=num_mo;i<num_global_atoms;i++)
        {
            fprintf(ftype, "%f\n", eigenvalues[i]*27.2);
            fmt::print(screen, "{}\n", eigenvalues[i]*27.2);
        }
        fclose(ftype);

        FILE *ftype2 = fopen("eigenvectors.dat", "w");
        fprintf(ftype2, "Occupied\n");
        for (int i=0;i<num_mo;i++)
        {
            for (int j=0;j<num_global_atoms;j++)
            {
                fprintf(ftype2, "%f ", coeff[i][j]);
            }
            fprintf(ftype2, "\n\n");
        }
        fprintf(ftype2, "Unoccupied\n");
        for (int i=0;i<10;i++)
        {
            std::string filename = "level" + std::to_string(i) + ".dat";
            const char* fname = filename.data();
            FILE *ftype3 = fopen(fname, "w");
            for (int j=0;j<num_global_atoms;j++)
            {
                fprintf(ftype3, "%f %f %f %f\n",
                relevant_atoms[j][0], relevant_atoms[j][1], relevant_atoms[j][2], coeff[num_mo-1-i][j]);
            }
            fclose(ftype3);
        }
        //for (int i=num_mo;i<num_global_atoms;i++)
        //{
        //    for (int j=0;j<num_global_atoms;j++)
        //    {
        //        fprintf(ftype2, "%f ", coeff[i][j]);
        //    }
        //    fprintf(ftype2, "\n\n");
        //}

        /*
        FILE *ftype3 = fopen("fock.dat", "w");
        for (int i=0;i<num_global_atoms;i++)
        {
            for (int j=0;j<num_global_atoms;j++)
            {
                fprintf(ftype3, "%f\n", fock[i][j]);
            }
        }
        fclose(ftype3);
        */
        fmt::print(screen, "{} {}\n", "Total energy:", total_energy*27.2);
        fmt::print(screen, "{}\n", "###############################################");
        fclose(ftype2);
    }
}

FixPPP::~FixPPP()
{
    post_mortem();
}
