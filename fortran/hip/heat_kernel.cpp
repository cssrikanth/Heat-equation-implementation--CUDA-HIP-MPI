#include <hip/hip_runtime.h>
#include <cstdio>
#include "utils.h"
#include "gpufort.h"

__global__ void  heat_eqn(
 double *t_d,
 double *t_old_d,                                                                    
 const int n1,                                                                            
 const int n2,                                                                            
 const int lb1,                                                                           
 const int lb2,                                                                           
 double r,                                                                                         
 int ny,                                                                                          
 int nx) {     
 int k = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
 #undef idx
 #define idx(a,b) ((a-(lb1))+n1*(b-(lb2)))
 if (k>lb1&&k<n1&&j>lb2&&j<n2) {
 t_d[idx(j,k)]=(t_old_d[idx(j,k)]+r*(t_old_d[idx((j+1),k)]+t_old_d[idx(j,(k+1))]+t_old_d[idx((j-1),k)]+t_old_d[idx(j,(k-1))]-4*t_old_d[idx(j,k)]));
 }
}

 
extern "C" void launch(
 const int sharedmem,
 hipStream_t stream,
 double *t_d,
 double *t_old_d,
 const int n1,
 const int n2,
 const int lb1,
 const int lb2,
 double r,
 int ny,
 int nx) {

 #define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

 const int blockX = 32; 
 const int blockY = 4; 

 dim3 block(blockX,blockY);

 const int nX = ny; 
 const int nY = nx; 

 const int gridX = divideAndRoundUp(nX,blockX);
 const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX,gridY);

 hipLaunchKernelGGL((heat_eqn), grid, block, sharedmem, stream, t_d,t_old_d,n1,n2,lb1,lb2,r,ny,nx);

}

__global__ void  swap_r1(
 double *t_d,
 const int td_n1,                                                                            
 const int td_n2,                                                                            
 const int td_lb1,                                                                           
 const int td_lb2,                                                                           
 double *tdr,
 const int tdr_n1,                                                                            
 const int tdr_n2,                                                                            
 const int tdr_lb1,                                                                           
 const int tdr_lb2,                                                                           
 int ny,                                                                                          
 int ng) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 #undef idx
 #define idx(a,b) ((a-(td_lb1))+td_n1*(b-(td_lb2)))
 #define idx_tr(a,b) ((a-(tdr_lb1))+tdr_n1*(b-(tdr_lb2)))
 if (j>td_lb2&&j<td_n2) {
 //printf("%d %d %d\n",ng,j,idx_tr(ng,j));
 printf("%f \n",t_d[idx(ng,j)]);
 t_d[idx(ng,j)]=tdr[idx_tr(ng,j)-1];
 }
}

extern "C" void launchr1(
 const int sharedmem,
 hipStream_t stream,
 double *t_d,
 const int td_n1,
 const int td_n2,
 const int td_lb1,
 const int td_lb2,
 double *tdr,
 const int tdr_n1,
 const int tdr_n2,
 const int tdr_lb1,
 const int tdr_lb2,
 int ny,
 int ng) {

 #define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

 const int blockX = 32; 
 //const int blockY = ng; 

 dim3 block(blockX);

 const int nX = ny; 
 //const int nY = ng; 

 const int gridX = divideAndRoundUp(nX,blockX);
 //const int gridY = divideAndRoundUp(nY,blockY);
 dim3 grid(gridX);

 hipLaunchKernelGGL((swap_r1), grid, block, sharedmem, stream, t_d,td_n1,td_n2,td_lb1,td_lb2,tdr,tdr_n1,tdr_n2,tdr_lb1,tdr_lb2,ny,ng);

}

__global__ void  swap_r2(
 double *t_d,
 const int td_n1,                                                                            
 const int td_n2,                                                                            
 const int td_lb1,                                                                           
 const int td_lb2,                                                                           
 double *tdr,
 const int tdr_n1,                                                                            
 const int tdr_n2,                                                                            
 const int tdr_lb1,                                                                           
 const int tdr_lb2,                                                                           
 int nx,                                                                                          
 int ny,                                                                                          
 int ng) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 #undef idx
 #define idx(a,b) ((a-(td_lb1))+td_n1*(b-(td_lb2)))
 #define idx_tr(a,b) ((a-(tdr_lb1))+tdr_n1*(b-(tdr_lb2)))
 if (j>td_lb2&&j<td_n2) {
 //printf("%d %d %d\n",ng,j,idx_tr(ng,j));
 t_d[idx(ng+nx+1,j)]=tdr[idx_tr(ng,j)-1];
 }
}

extern "C" void launchr2(
 const int sharedmem,
 hipStream_t stream,
 double *t_d,
 const int td_n1,
 const int td_n2,
 const int td_lb1,
 const int td_lb2,
 double *tdr,
 const int tdr_n1,
 const int tdr_n2,
 const int tdr_lb1,
 const int tdr_lb2,
 int nx,
 int ny,
 int ng) {

 #define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

 const int blockX = 32; 
 //const int blockY = 4; 

 dim3 block(blockX);

 const int nX = ny; 
 //const int nY = ng; 

 const int gridX = divideAndRoundUp(nX,blockX);
 //const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX);

 hipLaunchKernelGGL((swap_r2), grid, block, sharedmem, stream, t_d,td_n1,td_n2,td_lb1,td_lb2,tdr,tdr_n1,tdr_n2,tdr_lb1,tdr_lb2,nx,ny,ng);

}

__global__ void  swap_s(
 double *t_d,
 const int td_n1,                                                                            
 const int td_n2,                                                                            
 const int td_lb1,                                                                           
 const int td_lb2,                                                                           
 double *tds1,
 const int tds1_n1,                                                                            
 const int tds1_n2,                                                                            
 const int tds1_lb1,                                                                           
 const int tds1_lb2,                                                                           
 double *tds2,
 const int tds2_n1,                                                                            
 const int tds2_n2,                                                                            
 const int tds2_lb1,                                                                           
 const int tds2_lb2,                                                                           
 int nx,                                                                                          
 int ny,
 int ng) {     
 int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
 #undef idx
 #define idx(a,b) ((a-(td_lb1))+td_n1*(b-(td_lb2)))
 #define idx_ts(a,b) ((a-(tds1_lb1))+tds1_n1*(b-(tds1_lb2)))
 if (j>td_lb2&&j<td_n2) {
 //printf("%d %d %d %d %d\n",ng+1,j,idx(ng+1,j),td_lb1,td_n2);
 //printf("%f %f\n",t_d[idx(ng+1,j)],t_d[idx(ng+nx,j)]);
 tds1[idx_ts(ng,j)-1]=t_d[idx(ng+1,j)];
 tds2[idx_ts(ng,j)-1]=t_d[idx(nx+ng,j)];
 }
}

extern "C" void launchs(
 const int sharedmem,
 hipStream_t stream,
 double *t_d,
 const int td_n1,
 const int td_n2,
 const int td_lb1,
 const int td_lb2,
 double *tds1,
 const int tds1_n1,
 const int tds1_n2,
 const int tds1_lb1,
 const int tds1_lb2,
 double *tds2,
 const int tds2_n1,
 const int tds2_n2,
 const int tds2_lb1,
 const int tds2_lb2,
 int nx,
 int ny,
 int ng) {

 #define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

 const int blockX = 32; 
 //const int blockY = 4; 

 dim3 block(blockX);

 const int nX = ny; 
 //const int nY = ny; 

 const int gridX = divideAndRoundUp(nX,blockX);
 //const int gridY = divideAndRoundUp(nY,blockY);

 dim3 grid(gridX);

 hipLaunchKernelGGL((swap_s), grid, block, sharedmem, stream, t_d,td_n1,td_n2,td_lb1,td_lb2,tds1,tds1_n1,tds1_n2,tds1_lb1,tds1_lb2,tds2,tds2_n1,tds2_n2,tds2_lb1,tds2_lb2,nx,ny,ng);

}
