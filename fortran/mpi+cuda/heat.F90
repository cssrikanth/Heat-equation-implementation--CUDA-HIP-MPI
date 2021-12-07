module run
#ifdef USE_CUDA
 use cudafor
#endif
 use mpi

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: dp = doubtype
 integer, parameter :: mpi_prec = mpi_real8

 real(dp), allocatable, dimension(:,:) :: T_d, T_old_d
 real(dp), allocatable, dimension(:,:) :: T, x, y
 
 integer    :: n, ntime
 real(dp)   :: dom_len, delta, dt, nu, r, sigma
 integer    :: iercuda

 !MPI
 integer, parameter :: ndims = 3
 integer, dimension(:), allocatable  :: nblocks
 logical, dimension(:), allocatable  :: pbc
 integer, dimension(mpi_status_size) :: istatus

 integer, dimension(:), allocatable :: ncoords
 integer :: mp_cart,mp_cartx,mp_carty,mp_cartz
 integer :: nrank,nproc,nrank_x, nrank_y, nrank_z
 integer :: ileftx,irightx,ilefty,irighty,ileftz,irightz
 integer :: iermpi, iercuda

#ifdef USE_CUDA
 attributes(device) :: T_d, T_old_d
#endif


end module

module mod_mpi
 contains
  subroutine start_mpi()
   use run
   implicit none
   
   call mpi_init(iermpi)
   call mpi_comm_rank(mpi_comm_world,nrank,iermpi)
   call mpi_comm_size(mpi_comm_world,nproc,iermpi)

  end subroutine
end module



module mod_heat
 contains
  subroutine heat_eqn()
   use run
   implicit none
   type(dim3) :: grid,tBlock
   integer    :: i,j,k
  
   T_d = T
   do i=1,ntime
    print*,"time_it:", i
    T_old_d = T_d
    !$cuf kernel do(2) <<<*,*>>>
    do j=1,n
     do k=1,n
      if(j .ne. 1 .and. k .ne. 1 .and. j .ne. n .and. k .ne. n) then
       T_d(i,j) = T_old_d(i,j) + r*(T_old_d(i+1,j)+T_old_d(i,j+1)+T_old_d(i-1,j)+T_old_d(i,j-1)-4*T_old_d(i,j))
      end if
     end do
    end do 
   end do
   !@cuf iercuda=cudaDeviceSynchronize()
   T = T_d
  
  end subroutine 

end module

program heat
 use run
 use mod_heat 

 implicit none

 integer ::  i, j, k, z
 real(dp) :: start, finish

 call start_mpi()

 open(unit=11,file='input.dat',form='formatted')

 read (11,*) n, sigma, nu, dom_len, ntime

 delta =  dom_len/real(n-1)

 dt = (sigma * delta**2)/nu

! Allocate variables 
 allocate(T(n,n))
 allocate(x(n,n))
 allocate(y(n,n))
 allocate(T_d(n,n))
 allocate(T_old_d(n,n))

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
