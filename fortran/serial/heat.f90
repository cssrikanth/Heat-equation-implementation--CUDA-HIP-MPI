program heat

      implicit none

      real*8, allocatable, dimension(:,:) :: T, T_old, x, y
      real*8 :: dom_len, delta, dt, nu, r, sigma
      integer :: n, i, j, k, ntime,z
      real*8 :: start, finish


      open(unit=11,file='input.dat',form='formatted')

      read (11,*) n, sigma, nu, dom_len, ntime

      delta =  dom_len/real(n-1)

      dt = (sigma * delta**2)/nu


      allocate(T(n,n))
      allocate(T_old(n,n))
      allocate(x(n,n))
      allocate(y(n,n))

call cpu_time(start)


      x(1,:) = 0.0
      x(n,:) = dom_len
      y(:,1) = 0.0
      y(:,n) = dom_len

      do i=2,n-1
           x(i,:) = x(i-1,:) + delta
           y(:,i) = y(:,i-1) + delta
      end do

!! Setting initial conditions
!
      do i=1,n
         do j=1,n
           if(x(i,j) <= 1.5 .and. x(i,j) >= 0.5 .and. y(i,j) <= 1.5 .and. y(i,j) >= 0.5) then
                   T(i,j) = 2.0
           else
                   T(i,j) = 1.0
           end if
         end do
      end do
!
      open(unit=17,file='int.dat',form='formatted')
      do i=1,n
        do j=1,n
                write(17,*)x(i,j),y(i,j),T(i,j)
        end do
      end do

! cfl 

r = (nu*dt)/delta**2

      do i=1,ntime
      print*,"time_it:", i
        T_old(:,:) = T(:,:)
        do j=2,n-1
          do k=2,n-1
                T(j,k) = T_old(j,k) + r*(T_old(j+1,k)+T_old(j,k+1)+T_old(j-1,k)+T_old(j,k-1)-4*T_old(j,k))
          end do
        end do
      end do

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
