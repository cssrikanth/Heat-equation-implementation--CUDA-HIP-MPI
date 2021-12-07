module precision
	implicit none

	integer, parameter :: sp = kind(0.0)
	integer, parameter :: dp = kind(0.0d0)

end module

module run
	use precision
	use cudafor

	contains

	subroutine host(T, ntime, r, n)
		real(dp), device, allocatable, dimension(:,:) :: T_old_d
		real(dp), managed, dimension(:,:) :: T
		type(dim3) :: grid , tBlock
		integer :: i
		real(dp), value :: r
		integer, value :: n, ntime
	        
		tBlock = dim3 (32,8,1)
	        grid = dim3(ceiling(real(n)/ tBlock%x), ceiling(real(n)/ tBlock%y), 1)
	
	        allocate(T_old_d(n,n))
	
		do i=1,ntime
		print*,"time_it:", i
			T_old_d = T
			call heat_equation <<<grid ,tBlock >>>(T, T_old_d, r, n)
	        end do

        end subroutine 

	attributes(global) subroutine heat_equation(T_d, T_old_d, r, ngrid)
		implicit none
		real(dp) :: T_d(:,:), T_old_d(:,:)
		real(dp), value :: r
		integer, value :: ngrid
		integer :: i, j

		i = (blockIdx%x-1)* blockDim%x + threadIdx%x
		j = (blockIdx%y-1)* blockDim%y + threadIdx%y

		if(i .ne. 1 .and. j .ne. 1 .and. i .ne. ngrid .and. j .ne. ngrid) then
	               	T_d(i,j) = T_old_d(i,j) + r*(T_old_d(i+1,j)+T_old_d(i,j+1)+T_old_d(i-1,j)+T_old_d(i,j-1)-4*T_old_d(i,j))
		end if

		call syncthreads()
	end subroutine

end module

program heat
	use precision
	use cudafor
	use run 

        implicit none

	integer :: n
	real(dp), managed, allocatable, dimension(:,:) :: T
	real(dp), allocatable, dimension(:,:) :: x, y
        real(dp) :: dom_len, delta, dt, nu, r, sigma
        integer ::  i, j, k, ntime,z
	real(dp) :: start, finish

        open(unit=11,file='input.dat',form='formatted')
  
        read (11,*) n, sigma, nu, dom_len, ntime
  
        delta =  dom_len/real(n-1)
  
        dt = (sigma * delta**2)/nu
  
call cpu_time(start)
 
        allocate(T(n,n))
        allocate(x(n,n))
        allocate(y(n,n))

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

! cfl 

        r = (nu*dt)/delta**2


        call host(T, ntime, r, n)

call cpu_time(finish)

	print*,"simulation completed!!!!"
	print*,"total time:", finish - start

        open(unit=9,file='soln.dat',form='formatted')
        
        do i=1,n
          do j=1,n
                  write(9,*)x(i,j),y(i,j),T(i,j)
          end do
        end do

end program
