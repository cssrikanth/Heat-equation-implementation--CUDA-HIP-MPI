module run
 use cudafor

 implicit none

 integer, parameter :: doubtype = selected_real_kind(15,307)  ! double precision
 
 integer, parameter :: dp = doubtype

 real(dp), device, allocatable, dimension(:,:) :: T_d, T_old_d
 real(dp), allocatable, dimension(:,:) :: T, x, y
 
 integer    :: n, ntime
 real(dp)   :: dom_len, delta, dt, nu, r, sigma
 integer    :: iercuda

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
    !$cuf kernel do(2) <<<grid,tBlock>>>
    do j=1,n
     do k=1,n
      if(j .ne. 1 .and. k .ne. 1 .and. j .ne. n .and. k .ne. n) then
       T_d(j,k) = T_old_d(j,k) + r*(T_old_d(j+1,k)+T_old_d(j,k+1)+T_old_d(j-1,k)+T_old_d(j,k-1)-4*T_old_d(j,k))
      end if
     end do
    end do 
   end do
   !@cuf iercuda=cudaDeviceSynchronize()
   T = T_d
  
  end subroutine 

end module

program heat
 use cudafor
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
