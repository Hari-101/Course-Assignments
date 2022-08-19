//NOTE-1
//To test a certain function remove its index from function and gradient definition
//For example:To test function-1,change: 
//function val=myfun1(x) ----> function val=myfun(x) and similarly for mygrad1(x)
//Also,we have to set n=10 for function-1,function-2 and n=2 for function-3,function-4.

//NOTE-2
//The optimal point of rosenbrock function is [1,1,1,1,1,1,1,1,1,1],so to maintain uniformity :
//Starting point has been considered as [2,2,2,2,2,2,2,2,2,2]' or [2,2]'.
//Other initial points can be checked for by changing the starting point declared within each function.

// Test function-1 10D Quadratic Function
// Starting point x0 = [1 1 1 1 ...]'
function val=myfun1(x)
    val=0.0
    n = 10
    for i = 1:n
        val = val+i*x(i)*x(i)
    end
endfunction

// Gradient of function 1
function g=mygrad1(x)
    n = 10
    g = zeros(n,1)
    for i = 1:n
        g(i) = 2*i*x(i)
    end    
endfunction

// Test Function-2 Rosenbrock function
function val=myfun2(x)
    n = 10
    val = 0
    for i = 1:n-1
        val=val+(100*(x(i+1)-x(i)^2)^2) + (1-x(i))^2
    end
endfunction
    
// Gradient of function 2
function g=mygrad2(x)
    n = 10
    g = zeros(10 ,1)
    g(1) = (-400*(x(2)-x(1)^2)*x(1)) - 2*(1-x(1))
    g(10) = 200*(x(10)-x(9)^2)
    for i = 2:n-1
        g(i) = (-400*(x(i+1)-x(i)^2)*x(i)) - 2*(1-x(i)) + 200*(x(i)-x(i-1)^2)
    end
endfunction

// Test Function-3
// Choose starting point x0 = [1 1]'
function val = myfun3(x)
    val = ((x(1)^2*(x(1)-10)^2)+(x(2)^2*(x(2)-10)^2))
endfunction

// Gradient of function 3
function g = mygrad3(x)
    g = [0 0]'
    for i =1:2
    g(i)= 4*x(i)*(x(i)-10)*(x(i)-5)
    end     
endfunction

// Test Function-4  Booth Function
// Choose starting point x0 = [1 1]'
function val=myfun(x)
    val = 0
    val=(x(1)+(2*x(2))-7)^2 + ((2*x(1)+x(2)-5)^2)
endfunction

// Gradient of function 4
function g=mygrad(x)
    g = [0 0]'
    g(1)= 2*(x(1)+(2*x(2))-7) + 4*((2*x(1))+x(2)-5)
    g(2)= 4*(x(1)+(2*x(2))-7) + 2*((2*x(1))+x(2)-5)    
endfunction

function m=modsq(x)
    n = 2                                      //Set n=10 for function-1 and function-2
    m = 0                                       //Set n=2 for function-1 and function-2
    g = zeros(n,1)
   for i = 1:n
        m = m + x(i)^2 
   end   
endfunction

function xu = find_xu(x,s)
    xu = x + (1000 * (s/(sqrt(modsq(s)))))   
endfunction

function [x,val] = steepest()
    n = 2                                       //Set n=10 for function-1 and function-2
    count = 0                                   //Set n=2 for function-1 and function-2
    x = ones(n,1)+1                           //Setting the initial point
    xl = zeros(n,1)+0.5
    val = myfun(x)
    prev_val = 2*val
    //while (modsq((x-xl))> 0.01)
    while ((prev_val - val)> 0.000001 )               //To satisfy first iteration check condition
    //while (modsq(mygrad(x)) > 0.01)
        xl = x
        prev_val = val
        count = count + 1
        s = - (mygrad(xl))
        xu = find_xu(xl,s)
        [x,val]=line_search(xl, xu)
         
        disp ('value of function at iteration',count,val)
        disp('Point reached',x)
    end      
    disp ('Steepest descent number of iterations',count)
    disp ('Steepest descent final point',x)
    disp ('Steepest descent final function value',val)
    disp ('Gradient at x',mygrad(x))
endfunction

function [x,val] = BFGS()
    n=2                                        //Set n=10 for function-1 and function-2               
    count=0                                     //Set n=2 for function-1 and function-2
    x_k = ones(n ,1)+1                          //Setting Initial point
    prev_val = myfun(x_k)
    s1 = - (mygrad(x_k))
    xu = find_xu(x_k,s1)
    [x_kp1,val] = line_search(x_k, xu)
    H_k = eye(n,n)                       //Initialising H matrix as identity matrix
    while ((prev_val - val)> 0.000001 )
    //while (modsq(mygrad(x_kp1)) > 0.01)
        count=count + 1
        prev_val = val
        del_k = x_kp1 - x_k
        gama_k = mygrad(x_kp1)-mygrad(x_k)
        H_kp1 = H_k + ((1 + ((gama_k'*H_k*gama_k)/(del_k'*gama_k)))*((del_k*del_k')/(del_k'*gama_k)))- (((del_k*gama_k'*H_k)+(H_k*gama_k*del_k'))/(del_k'*gama_k))
        s_k = -(H_kp1*mygrad(x_kp1))
        xu = find_xu(x_kp1,s_k)
        [x,val] = line_search(xu, x_kp1)
        x_k = x_kp1
        x_kp1 = x
        disp ('value of function at iteration',count,val)
        disp('Point reached',x)
    end
    disp ('BFGS number of iterations',count)
    disp ('BFGS final point',x)
    disp ('BFGS final function value',val)
    disp ('Gradient at x',mygrad(x))
endfunction
 
function [x,val] = Fletcher_Reeves()
   count = 0
   n=2                                          //Set n=10 for function-1 and function-2
   conj_comp = zeros(n,1)                       //Set n=2 for function-1 and function-2
   xl = ones(n,1)+1                             //Setting the initial point
   prev_val = myfun(xl)
   g_i = (mygrad(xl))
   s_k = -(g_i)   
   xu = find_xu(xl,s_k)
   [x,val]=line_search(xl, xu)
   xl = x
   while ((prev_val - val)> 0.000001 )
   //while (modsq(mygrad(xl)) > 0.01 )        
        count = count + 1
        prev_val = val        
        g_ip1 = (mygrad(xl)) 
        bta = (modsq(g_ip1)/modsq(g_i))
        conj_comp = conj_comp + (bta*s_k)
        s_kp1 = -(g_ip1) + conj_comp
        xu = find_xu(xl,s_kp1)
        [x,val]=line_search(xl,xu)
        xl = x
        s_k = s_kp1
        g_i = g_ip1
        disp ('value of function at iteration',count,val)
        disp('Point reached',x)
   end 
   disp ('FR number of iterations',count)
   disp ('FR final point',x)
   disp ('FR final function value',val)
   disp ('Gradient at x',mygrad(x))
   
   adfs=0  // implement here
endfunction

function [x,val]=line_search(xl, xu)
    n = 10                                  //Set n=10 for function-1 and function-2
    line_count = 1                         //Set n=2 for function-1 and function-2
    s = xu - xl
    x3 = xl + (0.618 * s)
    fx3 = myfun(x3)
    x4 = xu - (0.618 * s)
    fx4 = myfun(x4)
    if  (fx3 < fx4)then xl = x4;val = fx3;x = x3;
        else xu = x3;val = fx4;x = x4; 
    end
    prev_val = 2 * val                      //To satisfy first iteration check condition
    while ((prev_val - val) > 0.000001 )
        line_count = line_count + 1
        prev_val = val
        s = xu - xl
        x3 = xl + (0.618 * s)
        fx3 = myfun(x3)
        x4 = xu - (0.618 * s)
        fx4 = myfun(x4)
        if  (fx3 < fx4)then xl = x4;val = fx3;x = x3;
            else xu = x3;val = fx4;x = x4;
        end
    end
endfunction
