/*** WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics) Code***/
/********* Created by Jamie MacLeod, University of Bristol *************/
/*** Force Calculation: On Simulating Free Surface Flows using SPH. Monaghan, J.J. (1994) ***/
/***			+ XSPH Correction (Also described in Monaghan) ***/
/*** Density Reinitialisation as in Colagrossi, A. and Landrini, M. (2003): Moving Least Squares***/
/*** Smoothing Kernel: Wendland's C2 ***/
/*** Integrator: Newmark-Beta ****/
/*** Variable Timestep Criteria: CFL + Monaghan, J.J. (1989) conditions ***/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <string.h>
#include <sstream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/LU>
#include <nanoflann.hpp>
#include <utils.h>
#include <KDTreeVectorOfVectorsAdaptor.h>

#ifndef M_PI
#define M_PI (4.0*atan(1.0))
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace nanoflann;

/*Make these from input file at some point...*/
//Simulation Definition
const static Vector2i xyPART(40,48); /*Number of particles in (x,y) directions*/
const static unsigned int SimPts = xyPART(0)*xyPART(1); /*total sim particles*/
static unsigned int bound_parts;			/*Number of boundary particles*/
static unsigned int npts;
const static double Pstep = 0.1;	/*Initial particle spacing*/

//Simulation window parameters
const static Vector2d Box(15,5); /*Boundary dimensions*/
const static Vector2d Start(0.11,0.11); /*Simulation particles start + end coords*/
const static Vector2d Finish(Start(0)+Pstep*xyPART(0)*1.0,Start(1)+Pstep*xyPART(1)*1.0);

//Fluid Properties
const static double rho0 = 1000.0; /*Rest density*/
const static double Simmass = rho0*((Finish(0)-Start(0))*(Finish(1)-Start(1))/(1.0*SimPts));
const static double Boundmass = 0.9*rho0*((Finish(0)-Start(0))*(Finish(1)-Start(1))/(1.0*SimPts));
const static double Cs = 50; /*Speed of sound*/
const static double gam = 7.0; /*Factor for Tait's Eq*/
const static double B = rho0*pow(Cs,2)/gam; /*Factor for Tait's Eq*/

// SPH Parameters
const static double H = sqrt((3/M_PI)*(Simmass/rho0)); /*Support Radius*/
const static double HSQ = H*H; 
const static double r0 = Pstep;		/*Boundary support radius*/
const static double D = pow(Cs,2);	/*Boundary param 1*/
const static float N1 = 4;			/*Boundary param 2*/
const static float N2 = 2;			/*Boundary param 3*/

//Timestep Parameters
static double dt = 0.0002;	/*Timestep*/
static float t = 0.f;				/*Current time in sim*/
const static float beta = 0.25;		/*Newmark-Beta parameter*/
const static int subits = 8;		/*Newmark-Beta iterations*/
const static int Nframe = 2000;		/*Number of output frames*/
const static int Ndtframe = 10; 	/*Timesteps per frame*/
static double maxmu = 0;
static double maxf = 0;
static double errsum = 0.0;
static double logbase = 0.0;
const static Vector2d zero(0.0,0.0);

typedef struct Particle {
Particle(Vector2d x, Vector2d v, Vector2d f, float rho, float Rrho, float m,bool bound)	:
	xi(x), v(v), V(0.0,0.0),  f(f), rho(rho), p(0.0), Rrho(Rrho), m(m), b(bound){}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Vector2d xi, v, V, f;
	float rho, p, Rrho, m;
	bool b;
	int size() const {return(xi.size());}
	double operator[](int a) const {
		return(xi[a]);
	}
}Particle;


typedef std::vector<Particle> State;
typedef KDTreeVectorOfVectorsAdaptor<State, double> my_kd_tree_t;
const static double search_radius = 4*HSQ;
nanoflann::SearchParams params;

/*Gaussian Smoothing Kernel, truncated at 3H */
double Kernel(double dist)
{
	double q = dist/H;
	if (q<3)
		return exp(-q*q)/(M_PI*HSQ);
	else 
		return 0;
}

/*Gaussian Gradient*/
Vector2d GradK(Vector2d Rij, double dist) 
{
	double q = dist/H;
	if (q<3)
		return 2*(Rij/HSQ)*exp(-q*q)/(M_PI*HSQ);
	else 
		return Vector2d(0,0);
}

/*Wendland's C2 Quintic Kernel*/
double W2Kernel(double dist) 
{
	double q = dist/H;
	if (q < 2)
		return (pow(1-0.5*q,4))*(2*q+1)*(7/(4*M_PI*HSQ));
	else
		return 0;
}

Vector2d W2GradK(Vector2d Rij, double dist)
{
	double q = dist/H;
	if (q < 2)
		return 5.0*(Rij/HSQ)*pow(1-0.5*q,3)*(7/(4*M_PI*HSQ));
	else
		return Vector2d(0.0,0.0);
}

void Forces(State &part,my_kd_tree_t &mat_index) 
{
	maxmu=0; 				/* CFL Parameter */
	double alpha = 0.025; 	/* Artificial Viscosity Parameter*/
	double eps = 0.2; 		/* XSPH Influence Parameter*/
	for (auto &pi :part) 
	{
		
		//Reset the particle to zero
		pi.f = zero;
		pi.V = pi.v;
		Vector2d contrib(0.0,0.0);
		double Rrhocontr = 0.0;
		pi.Rrho=0.0;
		
		vector<double> mu;
		mu.emplace_back(0);	/*Avoid deference of empty vector*/
		std::vector<std::pair<size_t,double>> matches; /* Nearest Neighbour Search*/
		mat_index.index->radiusSearch(&pi.xi[0], search_radius, matches, params);
		for (auto &i: matches) 
		{
			Particle pj = part[i.first];
			// if(&pi == &pj)
			// 	continue;

			Vector2d Rij = pj.xi-pi.xi;
			Vector2d Vij = pj.v-pi.v;
			double r = Rij.norm();
			double Kern = W2Kernel(r);
			Vector2d Grad = W2GradK(Rij, r);
			
			// if (pj.b==false) {
			/*Pressure and artificial viscosity calc - Monaghan 1994 p.400*/
				double cbar= 0.5*(sqrt((B*gam)/pi.rho)+sqrt((B*gam)/pj.rho));
				double vdotr = Vij.dot(Rij);
				double rhoij = 0.5*(pi.rho+pj.rho);
				double muij= H*vdotr/(r*r+0.01*HSQ);
				mu.emplace_back(muij);
				double pifac = alpha*cbar*muij/rhoij;

				if (vdotr > 0) pifac = 0;
				contrib += pj.m*Grad*(pifac - pi.p/pow(pi.rho,2)- pj.p/pow(pj.rho,2));
			
			//}
			// if (pj.b == true && r < r0) 
			// {
			// 	contrib -= D*(pow((r0/r),N1)-pow((r0/r),N2))*Rij/Rij.squaredNorm();
			// }
			//if (pj.b==false)
			
			pi.V+=eps*(pj.m/rhoij)*Kern*Vij; /* XSPH Influence*/
			Rrhocontr -= pj.m*(Vij.dot(Grad));
			
			
		}
		pi.Rrho = Rrhocontr; /*drho/dt*/
		pi.f= contrib;
		pi.f(1) += -9.81; /*Add gravity*/
		
		//CFL f_cv Calc
		double it = *max_element(mu.begin(),mu.end());
		if (it > maxmu)
			maxmu=it;

		//cout << pi.f << endl;
	}

}

/*Density Reinitialisation using Least Moving Squares as in A. Colagrossi (2003)*/
void DensityReinit(State &p, my_kd_tree_t &mat_index)
{
	Vector3d one(1.0,0.0,0.0);
	for(auto &pi: p)
	{
		Matrix3d A= Matrix3d::Zero();
		//Find matrix A.
		std::vector<std::pair<size_t,double>> matches;
		mat_index.index->radiusSearch(&pi.xi[0], search_radius, matches, params);
		for (auto &i: matches) 
		{
			Particle pj = p[i.first];
			Vector2d Rij = pi.xi-pj.xi;
			Matrix3d Abar;	
			Abar << 1      , Rij(0)        , Rij(1)        ,
				    Rij(0) , Rij(0)*Rij(0) , Rij(1)*Rij(0) ,
				    Rij(1) , Rij(1)*Rij(0) , Rij(1)*Rij(1) ;

			A+= W2Kernel(Rij.norm())*Abar*pj.m/pj.rho;
		}
		
		//Check if A is invertible
		Vector3d Beta;
		FullPivLU<Matrix3d> lu(A);
		if (lu.isInvertible())
			Beta = lu.inverse()*one;
		else
			Beta = (1/A(0,0))*one;

		//Find corrected kernel
		double rho = 0.0;
		for (auto &i: matches)
		{
			Vector2d Rij = pi.xi-p[i.first].xi;
			rho += p[i.first].m*W2Kernel(Rij.norm())*(Beta(0)+Beta(1)*Rij(0)+Beta(2)*Rij(1));
		}

		pi.rho = rho;
	}
}

void Newmark_Beta(State &pn, State &pnp1, my_kd_tree_t &mat_index) 
{
	vector<Vector2d> xih;
	for (auto pi :pnp1)
		xih.emplace_back(pi.xi);

	for (int k = 0; k < subits; ++k)
	{	
		Forces(pnp1, mat_index); /*Guess force at time n+1*/
		for (size_t i=0; i < bound_parts; ++i)
		{
			pnp1[i].f = zero; /*Zero boundary forces and velocities...*/
			pnp1[i].v = zero;  /*So that they dont move*/
			pnp1[i].V = zero;	
		}

		/*Previous State for error calc*/
		for (size_t  i=0; i< xih.size(); ++i)
			xih[i] = pnp1[i].xi;

		/*Update the state at time n+1*/
		for (size_t i=0; i < pn.size() ; ++i )
		{
			pnp1[i].v = pn[i].v+0.5*dt*(pn[i].f+pnp1[i].f);
			pnp1[i].rho = pn[i].rho+0.5*dt*(pn[i].Rrho+pnp1[i].Rrho);
			pnp1[i].xi = pn[i].xi+dt*pn[i].V+0.5*(dt*dt)*(1-2*beta)*pn[i].f+(dt*dt*beta)*pnp1[i].f;
			pnp1[i].p = B*(pow(pnp1[i].rho/rho0,gam)-1);
		}
		mat_index.index->buildIndex();
		
		errsum = 0.0;
		for (size_t i=0; i < pnp1.size(); ++i)
		{
			Vector2d r = pnp1[i].xi-xih[i];
			errsum += r.squaredNorm();
		}

		if(k == 0)
			logbase=log10(sqrt(errsum/(1.0*npts)));
	}

	
	/*Find maximum safe timestep*/
	vector<Particle>::iterator maxfi = std::max_element(pnp1.begin(),pnp1.end(),
		[](Particle p1, Particle p2){return p1.f.norm()< p2.f.norm();});
	maxf = maxfi->f.norm();
	double dtf = sqrt(H/maxf);
	double dtcv = H/(Cs+maxmu);
	dt = 0.3*min(dtf,dtcv);

	//Update the state at time n
	pn = pnp1;
	t+=dt;
}


void InitSPH(State &particles) 
{
	cout << "Initialising simulation with " << SimPts << " particles" << endl;
	
	//Structure input initialiser
	//Particle(Vector2d x, Vector2d v, Vector2d vh, Vector2d f, float rho, float Rrho, bool bound) :
	 
	Vector2d v(0.0,0.0);  
	Vector2d f(0.0,0.0); 
	float rho=rho0; 
	float Rrho=0.0; 

	/*create the boundary particles*/ 	 
	static double stepx = r0*0.5;
	static double stepy = r0*0.5;
	const static int Ny = ceil(Box(1)/stepy);
	stepy = Box(1)/Ny;
	const static int Nx = ceil(Box(0)/stepx);
	stepx = Box(0)/Nx;
	
	for(int i = 0; i <= Ny ; ++i) {
		Vector2d xi(0.f,i*stepy);
		particles.emplace_back(Particle(xi,v,f,rho,Rrho,Boundmass,true));
	}
	for(int i = 1; i <Nx ; ++i) {
		Vector2d xi(i*stepx,Box(1));
		particles.emplace_back(Particle(xi,v,f,rho,Rrho,Boundmass,true));
	}
	for(int i= Ny; i>0; --i) {
		Vector2d xi(Box(0),i*stepy);
		particles.emplace_back(Particle(xi,v,f,rho,Rrho,Boundmass,true));
	}
	for(int i = Nx; i > 0; --i) {
		Vector2d xi(i*stepx,0.f);
		particles.emplace_back(Particle(xi,v,f,rho,Rrho,Boundmass,true));
	}
	bound_parts = particles.size();
	
	/*Create the simulation particles*/
	for( int i=0; i< xyPART(0); ++i) 
	{
		for(int j=0; j< xyPART(1); ++j)
		{				
				Vector2d xi(Start(0)+i*Pstep,Start(1)+j*Pstep);		
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,Simmass,false));
		}
	}

	cout << "Total Particles: " << SimPts + bound_parts << endl;
}

void write_settings()
{
	std::ofstream fp("Test_Settings.txt", std::ios::out);

  if(fp.is_open()) {
    //fp << "VERSION: " << VERSION_TAG << std::endl << std::endl; //Write version
    fp << "SIMULATION PARAMETERS:" << std::endl; //State the system parameters.
    fp << "\tNumber of frames (" << Nframe <<")" << std::endl;
    fp << "\tSteps per frame ("<< Ndtframe << ")" << std::endl;
    fp << "\tTime step (" << dt << ")" << std::endl;
    fp << "\tParticle Spacing ("<< Pstep << ")" << std::endl;
    fp << "\tParticle Mass ("<< Simmass << ")" << std::endl;
    fp << "\tReference density ("<< rho0 << ")" << std::endl;
    fp << "\tSupport Radius ("<< H << ")" << std::endl;
    fp << "\tGravitational strength ("<< 9.81 << ")" << std::endl;
    fp << "\tNumber of boundary points (" << bound_parts << ")" << std::endl;
    fp << "\tNumber of simulation points (" << SimPts << ")" << std::endl;
    fp << "\tIntegrator type (Newmark_Beta)" << std::endl;

    fp.close();
  }
  else {
    cout << "Error opening the output file." << endl;
    exit(-1);
  }
}

void write_frame_data(State particles, std::ofstream& fp)
{
    fp <<  "ZONE t=\"Boundary Data\", STRANDID=1, SOLUTIONTIME=" << t << std::endl;
  	for (auto b=particles.begin(); b!=std::next(particles.begin(),bound_parts); ++b)
	{
        fp << b->xi(0) << " " << b->xi(1) << " ";
        fp << b->v.norm() << " ";
        fp << b->f.norm() << " ";
        fp << b->rho << " "  << b->p << std::endl;
  	}

    fp <<  "ZONE t=\"Particle Data\", STRANDID=2, SOLUTIONTIME=" << t << std::endl;
  	for (auto p=std::next(particles.begin(),bound_parts); p!=particles.end(); ++p)
	{
		//Eigen::Vector2d a=  p->f/p->rho;
		if (p->xi!=p->xi || p->v!=p->v || p->f!=p->f) {
			cerr << endl << "Simulation is broken. A value is nan." << endl;
			cerr << "Broken line..." << endl;
			cerr << p->xi(0) << " " << p->xi(1) << " ";
	        cerr << p->v.norm() << " ";
	        cerr << p->f.norm() << " ";
	        cerr << p->rho << " " << p->p << std::endl; 
	        fp.close();
			exit(-1);
		}
        fp << p->xi(0) << " " << p->xi(1) << " ";
        fp << p->v.norm() << " ";
        fp << p->f.norm() << " ";
        fp << p->rho << " "  << p->p << std::endl; 
  	}
}

int main(int argc, char *argv[]) 
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	//initialise the vector<Particle> particles
	State particles;
	State particlesh;

	/*Initialise particles*/
	InitSPH(particles);
	npts = bound_parts + SimPts;
	for (auto p: particles)
			particlesh.emplace_back(p);
	
	my_kd_tree_t mat_index(2,particlesh,10);
	mat_index.index->buildIndex();

	Forces(particles, mat_index);
	for (size_t i=0; i < bound_parts; ++i)
		particles[i].f = zero;

	/*Write settings*/
	write_settings();

	/*Open simulation files*/
	std::ofstream f1("Test.plt", std::ios::out);
	
	if (f1.is_open())
	{
		f1 << std::fixed << setprecision(3);
		
		cout << "Starting simulation..." << endl;
		//Write file header defining veriable names
		f1 <<  "VARIABLES = x, y, V, F, rho, P" << std::endl;
		write_frame_data(particles, f1);
		
		for (int frame = 1; frame<= Nframe; ++frame) {
				  cout << "Frame Number: " << frame << "\tError: " << log10(sqrt(errsum/(1.0*npts)))-logbase << endl;
				  for (int i=0; i< Ndtframe; ++i) {
				    Newmark_Beta(particles, particlesh, mat_index);
				  }
				  DensityReinit(particlesh,mat_index);
				  write_frame_data(particlesh, f1);
				}
	f1.close();
	
	}
	else
	{
		cerr << "Error opening frame output file." << endl;
		exit(-1);
	}


	cout << "Simulation complete. Output is available in /Output!" << endl;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2-t1).count();
    cout << "Time taken:\t" << duration/1e06 << " seconds" << endl;
	return 0;

}
