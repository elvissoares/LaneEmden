#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>

#include "cpp_constants.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

int func (double t, const double y[], double f[], void *params) {
	double mu = *(double *)params;
	
	f[0] = y[1] / Q(t);
	f[1] = - pow(y[0], mu) * Q(t);
	return GSL_SUCCESS;
}

int jac (double t, const double y[], double *dfdy, double dfdt[], void *params) {
	// get parameter(s) from params; here, just a double 
	double mu = *(double *)params;
	
	gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 2, 2);
	gsl_matrix * m_ptr = &dfdy_mat.matrix; // m_ptr points to the matrix 
	
	// fill the Jacobian matrix as shown 
	gsl_matrix_set (m_ptr, 0, 0, 0.0); 	// df[0]/dy[0] 
	gsl_matrix_set (m_ptr, 0, 1, 1./Q(t));	// df[0]/dy[1] 
	gsl_matrix_set (m_ptr, 1, 0, -mu*pow(y[0], mu-1)*Q(t));	// df[1]/dy[0]
	gsl_matrix_set (m_ptr, 1, 1, 0.0);	// df[1]/dy[1]
	
	// set explicit t dependence of f[i] 
	dfdt[0] = -2.*y[1]/C(t);
	dfdt[1] = -2.* pow(y[0], mu) * t;
	
	return GSL_SUCCESS;	// GSL_SUCCESS defined in gsl/errno.h as 0 
}


int main (void){
	
	double n = 4/3.;
	
	gsl_odeiv2_system sys = {func, jac, 2, &n};
    gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rk8pd, 1e-6, 1e-6, 0.0);

    double csi = 1e-4;
    
    double theta_i = 1 - Q(csi)/6. + n * pow(csi, 4)/120. ;
    double dtheta_i = - csi/3. + n * pow(csi, 3)/30. ;
    
    double y[2] = {theta_i, Q(csi)*dtheta_i};
    
    std::cout << "0" << "  " << "1" << "  " << "0" << std::endl;
     
    while ( y[0] > 0. ){ 
    //for (unsigned int i = 1; i <= 100; i++){
		
		double csi_i = csi + 1e-5;
        int status = gsl_odeiv2_driver_apply (d, &csi, csi_i, y);
     
        if (status != GSL_SUCCESS){
     	  std::cerr << "error, return value=" << status << std::endl;
     	  break;
     	}
     
        std::cout <<  csi << "  " << y[0] << "  " << y[1] << std::endl;
    }
     
    gsl_odeiv2_driver_free (d);
    return 0;
}
