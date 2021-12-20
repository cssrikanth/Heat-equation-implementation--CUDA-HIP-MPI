#ifdef __GPUFORT
#include "heat.F90-fort2hip.f08"
#endif
module run
#ifdef __GPUFORT
 use hipfort
 use hipfort_check
#else
 use cudafor
#endif

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: dp = doubtype

#ifdef __GPUFORT
 real(dp),pointer,dimension(:,:) :: t_d
 real(dp),pointer,dimension(:,:) :: t_old_d
#else
 real(dp), device, allocatable, dimension(:,:) :: T_d, T_old_d
#endif
 real(dp), allocatable, dimension(:,:) :: T, x, y
 
 integer    :: n, ntime
 real(dp)   :: dom_len, delta, dt, nu, sigma
 real :: r
 integer    :: iercuda

end module

module mod_heat
 contains
  subroutine heat_eqn()
#ifdef __GPUFORT
   use mod_heat_fort2hip
#endif
   use run
   implicit none
   type(dim3) :: grid,tBlock
   integer    :: i,j,k
   tBlock = dim3(128,2,1)
   grid = dim3(ceiling(real(n-(1)+1))/tBlock%x,ceiling(real(n-(1)+1))/tBlock%y,1)

  
#ifdef __GPUFORT
   call hipCheck(hipMemcpy(T_d, T, hipMemcpyHostToDevice))
#else
   T_d = T
#endif
   do i=1,ntime
    print*,"time_it:", i
#ifdef __GPUFORT
    call hipCheck(hipMemcpy(T_old_d, T_d, hipMemcpyDeviceToDevice))
    ! extracted to HIP C++ file
    call launch_heat_eqn_35_02ced1(grid,tBlock,0,c_null_ptr,n,c_loc(t_d),size(t_d,1),size(t_d,2),lbound(t_d,1),lbound(t_d,2),c_loc(t_old_d),size(t_old_d,1),size(t_old_d,2),lbound(t_old_d,1),lbound(t_old_d,2),r)
#else
    T_old_d = T_d
    !$cuf kernel do(2) <<<grid,tBlock>>>
    do j=1,n
     do k=1,n
      if(j .ne. 1 .and. k .ne. 1 .and. j .ne. n .and. k .ne. n) then
       T_d(j,k) = T_old_d(j,k) + r*(T_old_d(j+1,k)+T_old_d(j,k+1)+T_old_d(j-1,k)+T_old_d(j,k-1)-4*T_old_d(j,k))
      end if
     end do
    end do 
#endif
   end do
   !@cuf iercuda=cudaDeviceSynchronize()
#ifdef __GPUFORT
   call hipCheck(hipMemcpy(T, T_d, hipMemcpyDeviceToHost))
#else
   T = T_d
#endif
  
  end subroutine 

end module

program heat
#ifdef __GPUFORT
 use hipfort
 use hipfort_check
#else
 use cudafor
#endif
 use run
 use mod_heat 

 implicit none

 integer ::  i, j, k, z
 real(dp) :: start, finish

 open(unit=11,file='input.dat',form='formatted')

 read (11,*) n, sigma, nu, dom_len, ntime

 delta =  dom_len/real(n-1)

 dt = (sigma * delta**2)/nu

! Allocate variables 
 allocate(T(n,n))
 allocate(x(n,n))
 allocate(y(n,n))
#ifdef __GPUFORT
 call hipCheck(hipMalloc(T_d, n,n))
 call hipCheck(hipMalloc(T_old_d, n,n))
#else
 allocate(T_d(n,n))
 allocate(T_old_d(n,n))
#endif

 x(1,:) = 0.0
 x(n,:) = dom_len
 y(:,1) = 0.0
 y(:,n) = dom_len

 do i=2,n-1
  x(i,:) = x(i-1,:) + delta
  y(:,i) = y(:,i-1) + delta
 end do

! Setting initial conditions

 do i=1,n
  do j=1,n
   if(x(i,j) <= 1.5 .and. x(i,j) >= 0.5 .and. y(i,j) <= 1.0 .and. y(i,j) >= 0.5) then
    T(i,j) = 2.0
   else
    T(i,j) = 1.0
   end if
  end do
 end do

 open(unit=17,file='int.dat',form='formatted')
 do i=1,n
  do j=1,n
   write(17,*)x(i,j),y(i,j),T(i,j)
  end do
 end do
 close(17)

! cfl 
 r = (nu*dt)/delta**2
 
 call cpu_time(start)

 call heat_eqn()

 call cpu_time(finish)
 
 open(unit=17,file='soln.dat',form='formatted')
 do i=1,n
  do j=1,n
   write(17,*)x(i,j),y(i,j),T(i,j)
  end do
 end do
 close(17)

 print*,"simulation completed!!!!"
 print*,"total time:", finish - start

end program