#include <hip/hip_runtime.h>
#include <cstdio>
#include <float.h>

#ifdef SINGLE_PRECISION
  typedef float real;
#else
  typedef double real;
#endif

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

//Access 2D arrays with ghost
#define idx(i,j) (((i)-1+ng)+(nx+2*ng)*((j)-1+ng))

//Access 2D arrays for swapping
#define idx_s(i,j) (((i)-1)+ng*((j)-1))

__global__ void  heat_eqn(
 real *t_d,
 real *t_old_d,                                                                    
 real r,                                                                                         
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int k = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 t_d[idx(j,k)]=(t_old_d[idx(j,k)]+r*(t_old_d[idx((j+1),k)]+t_old_d[idx(j,(k+1))]+t_old_d[idx((j-1),k)]+t_old_d[idx(j,(k-1))]-4*t_old_d[idx(j,k)]));
}

 
extern "C" void launch(
 real *t_d,
 real *t_old_d,
 real r,
 int ng,
 int ny,
 int nx) {

 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ny; 
 const int nY = nx; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);

 hipLaunchKernelGGL((heat_eqn),grid,block,0,0,t_d,t_old_d,r,ng,ny,nx);

}

__global__ void  swap_r1(
 real *t_d,
 real *tdr,
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int k = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 t_d[idx(j-ng,k)]=tdr[idx_s(j,k)];
}

extern "C" void launchr1(
 real *t_d,
 real *tdr,
 int ng,
 int ny,
 int nx) {

 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ng; 
 const int nY = ny; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);
 
 hipLaunchKernelGGL((swap_r1), grid, block, 0,0, t_d,tdr,ng,ny,nx);

}

__global__ void  swap_r2(
 real *t_d,
 real *tdr,
 int ng,                                                                                          
 int ny,                                                                                          
 int nx) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int k = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 t_d[idx(nx+j,k)]=tdr[idx_s(j,k)];
}

extern "C" void launchr2(
 real *t_d,
 real *tdr,
 int ng,
 int ny,
 int nx) {
 
 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ng; 
 const int nY = ny; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);

 hipLaunchKernelGGL((swap_r2),grid,block,0,0,t_d,tdr,ng,ny,nx);

}

__global__ void  swap_s(
 real *t_d,
 real *tds1,
 real *tds2,
 int ng,                                                                                          
 int ny,
 int nx) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int k = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 tds1[idx_s(j,k)]=t_d[idx(j,k)];
 tds2[idx_s(j,k)]=t_d[idx(nx-ng+j,k)];
}

extern "C" void launchs(
 real *t_d,
 real *tds1,
 real *tds2,
 int ng,
 int ny,
 int nx) {
 
 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ng; 
 const int nY = ny; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);

 hipLaunchKernelGGL((swap_s),grid,block,0,0,t_d,tds1,tds2,ng,ny,nx);

}
