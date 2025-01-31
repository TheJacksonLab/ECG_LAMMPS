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

#include "atom.h"
#include "comm.h"
#include "text_file_reader.h"
#include "error.h"
#include "domain.h"
#include "math_eigen_impl.h"

using namespace LAMMPS_NS;

#include "GPUSolver.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cusolver_common.h>

GPUSolver::GPUSolver()
{
    fmt::print(stdout, "\n{}\n", "this works");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp propobject;
    fmt::print(stdout, "{} {}\n", "Number of CUDA-capable devices: ", deviceCount); 
    for (int i=0;i<deviceCount;i++)
    {
       cudaGetDeviceProperties(&propobject, i);
       fmt::print(stdout, "{} {}\n", "Name: ", propobject.name); 
    }
    // Test solvers
    double *test_mat;
    test_mat = new double [4];
    test_mat[0] = 1;
    test_mat[1] = -1;
    test_mat[2] = -1;
    test_mat[3] = 1;

    // Allocate memory on gpu for the matrix
    double *test_mat_gpu;
    cudaError_t status = cudaMalloc(&test_mat_gpu, 4*sizeof(double));
    const char* status_name = cudaGetErrorName(status);
    fmt::print(stdout, "{} {}\n", "cudaMalloc:", status_name);

    // Copy the CPU matrix to the GPU
    cudaError_t copy_status = cudaMemcpy(test_mat_gpu, test_mat, 4*sizeof(double), cudaMemcpyHostToDevice);
    const char *copy_status_name = cudaGetErrorName(copy_status);
    fmt::print(stdout, "{} {}\n", "cudaMemcpy:", copy_status_name);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t handle_status = cusolverDnCreate(&cusolverH);
    if (handle_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnCreate success");
    }

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int lwork=0;
    int matsize=2;
    double *device_eigenvalues;

    cusolverStatus_t buffer_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, matsize, test_mat_gpu, matsize, device_eigenvalues, &lwork);
    if (buffer_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnDsyevd_bufferSize success");
       fmt::print(stdout, "{} {}\n", "Buffer size:", lwork);
    }

    // allocate memory for eigenvalues
    cudaError_t eigen_alloc = cudaMalloc(&device_eigenvalues, 2*sizeof(double));
    const char* eigen_alloc_name = cudaGetErrorName(eigen_alloc);
    fmt::print(stdout, "{} {}\n", "cudaMallocEigenvalue: ", eigen_alloc_name);

    // allocate work space for eigendecomposition
    double *device_work;
    cudaError_t work_alloc = cudaMalloc(&device_work, lwork*sizeof(double));
    const char* work_alloc_name = cudaGetErrorName(work_alloc);
    fmt::print(stdout, "{} {}\n", "cudaMallocWork: ", work_alloc_name);

    int *device_info;
    cudaError_t info_status = cudaMalloc(&device_info, sizeof(int));
    const char* info_status_name = cudaGetErrorName(info_status);
    fmt::print(stdout, "{} {}\n", "cudaMallocInfo: ", info_status_name);

    // calculate spectrum
    cusolverStatus_t solver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, matsize, 
                      test_mat_gpu, matsize, device_eigenvalues, device_work, lwork, device_info);
    if (solver_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnDsyevd success");
    }

    // copy eigenvalues to CPU
    double *eigenvalues;
    eigenvalues = new double[2];
    cudaError_t copy_eigen_back = cudaMemcpy(eigenvalues, device_eigenvalues, 2*sizeof(double), cudaMemcpyDeviceToHost);
    const char* copy_eigen_back_name = cudaGetErrorName(copy_eigen_back);
    fmt::print(stdout, "{} {}\n", "cudaMemcpyEigenBack:", copy_eigen_back_name);

    // print eigenvalues
    for (int i=0;i<2;i++)
    {
       fmt::print(stdout, "{} ", eigenvalues[i]); 
    }
    fmt::print(stdout, "{}", "\n");

    // copy eigenvectors to CPU
    cudaError_t copy_eigvec_back = cudaMemcpy(test_mat, test_mat_gpu, 4*sizeof(double), cudaMemcpyDeviceToHost);
    const char* copy_eigvec_back_name = cudaGetErrorName(copy_eigvec_back);
    fmt::print(stdout, "{} {}\n", "cudaMemcpyEigenvec:", copy_eigvec_back_name);


    // print eigenvectors
    for (int i=0;i<2;i++)
    {
      fmt::print(stdout, "{} {}\n", "Eigenvector ", i);
      for (int j=0;j<2;j++)
      {
         fmt::print(stdout, "{} ", test_mat[i*2 + j]);
      }
      fmt::print(stdout, "{}", "\n");
    }
}

void GPUSolver::solveFockMatrixCUDA(double *fock, int matsize, double *eigenvalues)
{

    // Allocate memory on gpu for the matrix
    double *fock_gpu;
    cudaError_t status = cudaMalloc(&fock_gpu, matsize*matsize*sizeof(double));
    const char* status_name = cudaGetErrorName(status);
    fmt::print(stdout, "{} {}\n", "cudaMallocFockMatrix:", status_name);

    // Copy the CPU matrix to the GPU
    cudaError_t copy_status = cudaMemcpy(fock_gpu, fock, matsize*matsize*sizeof(double), cudaMemcpyHostToDevice);
    const char *copy_status_name = cudaGetErrorName(copy_status);
    fmt::print(stdout, "{} {}\n", "cudaMemcpyFockMatrix:", copy_status_name);

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t handle_status = cusolverDnCreate(&cusolverH);
    if (handle_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnCreate success");
    }

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int lwork=0;
    double *device_eigenvalues;

    cusolverStatus_t buffer_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, matsize, 
               fock_gpu, matsize, device_eigenvalues, &lwork);
    if (buffer_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnDsyevd_bufferSize success");
       fmt::print(stdout, "{} {}\n", "Buffer size:", lwork);
    }

    // allocate memory for eigenvalues
    cudaError_t eigen_alloc = cudaMalloc(&device_eigenvalues, matsize*sizeof(double));
    const char* eigen_alloc_name = cudaGetErrorName(eigen_alloc);
    fmt::print(stdout, "{} {}\n", "cudaMallocEigenvalue: ", eigen_alloc_name);

    // allocate work space for eigendecomposition
    double *device_work;
    cudaError_t work_alloc = cudaMalloc(&device_work, lwork*sizeof(double));
    const char* work_alloc_name = cudaGetErrorName(work_alloc);
    fmt::print(stdout, "{} {}\n", "cudaMallocWork: ", work_alloc_name);

    int *device_info;
    cudaError_t info_status = cudaMalloc(&device_info, sizeof(int));
    const char* info_status_name = cudaGetErrorName(info_status);
    fmt::print(stdout, "{} {}\n", "cudaMallocInfo: ", info_status_name);

    // calculate spectrum
    cusolverStatus_t solver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, matsize, 
                      fock_gpu, matsize, device_eigenvalues, device_work, lwork, device_info);
    if (solver_status == CUSOLVER_STATUS_SUCCESS)
    {
       fmt::print(stdout, "{}\n", "cusolverDnDsyevd success");
    }

    // copy eigenvalues to CPU
    cudaError_t copy_eigen_back = cudaMemcpy(eigenvalues, device_eigenvalues, matsize*sizeof(double), cudaMemcpyDeviceToHost);
    const char* copy_eigen_back_name = cudaGetErrorName(copy_eigen_back);
    fmt::print(stdout, "{} {}\n", "cudaMemcpyEigenBack:", copy_eigen_back_name);

    // print eigenvalues
    
    //for (int i=0;i<2;i++)
    //{
    //   fmt::print(stdout, "{} ", eigenvalues[i]); 
    //}
    //fmt::print(stdout, "{}", "\n");
    

    // copy eigenvectors to CPU
    cudaError_t copy_eigvec_back = cudaMemcpy(fock, fock_gpu, matsize*matsize*sizeof(double), cudaMemcpyDeviceToHost);
    const char* copy_eigvec_back_name = cudaGetErrorName(copy_eigvec_back);
    fmt::print(stdout, "{} {}\n", "cudaMemcpyEigenvec:", copy_eigvec_back_name);

    cudaFree(device_eigenvalues);
    cudaFree(fock_gpu);
    cudaFree(device_work);
    cusolverDnDestroy(cusolverH);


    // print eigenvectors
    /*
    for (int i=0;i<2;i++)
    {
      fmt::print(stdout, "{} {}\n", "Eigenvector ", i);
      for (int j=0;j<2;j++)
      {
         fmt::print(stdout, "{} ", test_mat[i*2 + j]);
      }
      fmt::print(stdout, "{}", "\n");
    }
    */
}

void GPUSolver::buildDensityMatrixCUDA(double *A, double *B, double *C, int num_global_atoms, int num_mo, double alpha, double beta)
{
   //create cublas handle
   cublasHandle_t cuhandle;
   cublasCreate(&cuhandle);
   
   // allocate memory on GPU for the matrices
   double *A_gpu, *B_gpu, *C_gpu;
   cudaError_t A_alloc = cudaMalloc(&A_gpu, num_global_atoms*num_mo*sizeof(double));
   if (A_alloc == cudaSuccess)
   {
      fmt::print(stdout, "{}\n", "cudaMallocA success");
   }
   cudaError_t B_alloc = cudaMalloc(&B_gpu, num_global_atoms*num_mo*sizeof(double));
   if (B_alloc == cudaSuccess)
   {
      fmt::print(stdout, "{}\n", "cudaMallocB success");
   }
   cudaError_t C_alloc = cudaMalloc(&C_gpu, num_global_atoms*num_global_atoms*sizeof(double));
   if (C_alloc == cudaSuccess)
   {
      fmt::print(stdout, "{}\n", "cudaMallocC success");
   }

   // copy A and B from CPU to GPU
   cudaError_t A_copy = cudaMemcpy(A_gpu, A, num_global_atoms*num_mo*sizeof(double), cudaMemcpyHostToDevice);
   cudaError_t B_copy = cudaMemcpy(B_gpu, B, num_global_atoms*num_mo*sizeof(double), cudaMemcpyHostToDevice);

   cublasOperation_t transA = CUBLAS_OP_T;
   cublasOperation_t transB = CUBLAS_OP_N;

   // do the dgemm
   cublasStatus_t mul = cublasDgemm(cuhandle, transA, transB, num_global_atoms, num_global_atoms, num_mo, 
                       &alpha, A_gpu, num_mo, B_gpu, num_mo, &beta, C_gpu, num_global_atoms);

   // copy it back to C
   cudaError_t C_copy = cudaMemcpy(C, C_gpu, num_global_atoms*num_global_atoms*sizeof(double), cudaMemcpyDeviceToHost);

   cudaFree(A_gpu);
   cudaFree(B_gpu);
   cudaFree(C_gpu);
   cublasDestroy(cuhandle);
}

double GPUSolver::calculateResidualMatrixCUDA(double *density, double *fock, double *residual, int matsize)
{
   cublasHandle_t cuhandle;
   cublasCreate(&cuhandle);

   double alpha = 1;
   double beta=0;
   double norm=0;
   //memory allocation
   double *density_gpu, *fock_gpu, *fd_gpu, *df_gpu, norm_gpu;
   cudaError_t dgpu = cudaMalloc(&density_gpu, matsize*matsize*sizeof(double));
   cudaError_t fgpu = cudaMalloc(&fock_gpu, matsize*matsize*sizeof(double));
   cudaError_t fdgpu = cudaMalloc(&fd_gpu, matsize*matsize*sizeof(double));
   cudaError_t dfgpu = cudaMalloc(&df_gpu, matsize*matsize*sizeof(double));
   //cudaError_t ngpu = cudaMalloc(&norm_gpu, sizeof(double));

   //copy to device
   cudaError_t dgpucopy = cudaMemcpy(density_gpu, density, matsize*matsize*sizeof(double), cudaMemcpyHostToDevice);
   cudaError_t fgpucopy = cudaMemcpy(fock_gpu, fock, matsize*matsize*sizeof(double), cudaMemcpyHostToDevice);

   cublasOperation_t transA = CUBLAS_OP_N;
   cublasOperation_t transB = CUBLAS_OP_N;

   alpha=-1;

   // calculate FD product
   cublasStatus_t fd_calculate = cublasDgemm(cuhandle, transA, transB, matsize, matsize, matsize,
                                 &alpha, fock_gpu, matsize, density_gpu, matsize, &beta, fd_gpu, matsize);

   cublasStatus_t df_calculate = cublasDgemm(cuhandle, transA, transB, matsize, matsize, matsize,
                                 &alpha, density_gpu, matsize, fock_gpu, matsize, &beta, df_gpu, matsize);

   int nelements = matsize*matsize;
   cublasStatus_t r_construct = cublasDaxpy(cuhandle, nelements, &alpha, df_gpu, 1, fd_gpu, 1);

   //calculate norm for error
   cublasStatus_t norm_get = cublasDnrm2(cuhandle, nelements, fd_gpu, 1, &norm_gpu);

   // copy fd_gpu to the CPU, and also norm
   cudaError_t copyback = cudaMemcpy(residual, fd_gpu, matsize*matsize*sizeof(double), cudaMemcpyDeviceToHost);

   cudaFree(density_gpu);
   cudaFree(fock_gpu);
   cudaFree(fd_gpu);
   cudaFree(df_gpu);
   cublasDestroy(cuhandle);

   return norm_gpu;
}

void GPUSolver::constructBMatrix(double **residual_vectors, int matsize, int ndiis, double *B)
{
   int bmatsize = ndiis+1;
   double *residual_vector_i, *residual_vector_j;
   double rij=0.0;
   cublasHandle_t cuhandle;
   cublasCreate(&cuhandle);
   cudaError_t vi_alloc = cudaMalloc(&residual_vector_i, matsize*matsize*sizeof(double));
   cudaError_t vj_alloc = cudaMalloc(&residual_vector_j, matsize*matsize*sizeof(double));
   for (int i=0;i<ndiis;i++)
   {
      cudaError_t copy_i = cudaMemcpy(residual_vector_i, residual_vectors[i], matsize*matsize*sizeof(double),
         cudaMemcpyHostToDevice);
      for (int j=i;j<ndiis;j++)
      {
         cudaError_t copy_j = cudaMemcpy(residual_vector_j, residual_vectors[j], matsize*matsize*sizeof(double),
         cudaMemcpyHostToDevice);
         cublasStatus_t rdot = cublasDdot(cuhandle, matsize*matsize, residual_vector_i, 1,  
         residual_vector_j, 1, &rij);
         B[i*(ndiis+1)+j] = rij;
         B[j*(ndiis+1)+i] = rij;
      }
   }
   cudaFree(residual_vector_i);
   cudaFree(residual_vector_j);
   for (int i=0;i<ndiis+1;i++)
   {
      B[i*(ndiis+1) + ndiis] = -1.0;
      B[ndiis*(ndiis+1) + i] = -1.0;
   }
   B[(ndiis+1)*(ndiis+1) - 1] = 0.0;

   cublasDestroy(cuhandle);
}

void GPUSolver::constructDIISFockMatrix(double *fock, double **solution_vectors, double *coeff, int ndiis, int matsize)
{
   double *fock_gpu, *solution_vector;

   cudaError_t falloc = cudaMalloc(&fock_gpu, matsize*matsize*sizeof(double));
   cudaError_t init =  cudaMemset(fock_gpu, 0.0, matsize*matsize*sizeof(double));
   cudaError_t salloc = cudaMalloc(&solution_vector, matsize*matsize*sizeof(double));

   cublasHandle_t cuhandle;
   cublasCreate(&cuhandle);

   for (int i=0;i<ndiis;i++)
   {
      cudaError_t scopy = cudaMemcpy(solution_vector, solution_vectors[i], matsize*matsize*sizeof(double), 
      cudaMemcpyHostToDevice);
      cublasStatus_t addtofock = cublasDaxpy(cuhandle, matsize*matsize, &coeff[i], solution_vector, 1, fock_gpu, 1);
   }
   cudaError_t copyback = cudaMemcpy(fock, fock_gpu, matsize*matsize*sizeof(double), cudaMemcpyDeviceToHost);

   cudaFree(fock_gpu);
   cudaFree(solution_vector);
   cublasDestroy(cuhandle);
}

double GPUSolver::addtwonumbers(double a, double b)
{
    return a + b;
}
