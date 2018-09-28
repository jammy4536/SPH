/*********   WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics) Code   *************/
/*********        Created by Jamie MacLeod, University of Bristol               *************/
/*** Force Calculation: On Simulating Free Surface Flows using SPH. Monaghan, J.J. (1994) ***/
/***			        + XSPH Correction (Also described in Monaghan)                    ***/
/*** Viscosity:         Laminar Viscosity as described by Morris                          ***/
/*** Surface Tension:   Tartakovsky and Panchenko (2016)                                  ***/
/*** Density Reinitialisation: Colagrossi, A. and Landrini, M. (2003): Moving Least Squares***/
/*** Smoothing Kernel: Wendland's C2 ***/
/*** Integrator: Newmark-Beta ****/
/*** Variable Timestep Criteria: CFL + Monaghan, J.J. (1989) conditions ***/

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <string.h>
#include <sstream>
#include <chrono>
#include "Eigen/Core"
#include "Eigen/StdVector"
#include "Eigen/LU"
#include "NanoFLANN/nanoflann.hpp"
#include "NanoFLANN/utils.h"
#include "NanoFLANN/KDTreeVectorOfVectorsAdaptor.h"
/*#include "TECIO.h" 
#include "MASTER.h"*/ /* for defintion of NULL */

#ifndef M_PI
#define M_PI (4.0*atan(1.0))
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace nanoflann;

/*Make these from input file at some point...*/
//Simulation Definition

static double maxmu = 0.0;			/*CFL Parameter*/
static double errsum = 0.0;
static double logbase = 0.0;

typedef struct SIM {
	Vector2i xyPART; 						/*Starting sim particles in x and y box*/
	unsigned int SimPts,bound_parts,npts;	/*Number of particles*/
	double Pstep,Bstep;						/*Initial spacings*/
	Vector2d Box;							/*Box dimensions*/
	Vector2d Start, Finish; 				/*Starting sim box dimensions*/
	unsigned int subits,Nframe; 			/*Timestep values*/
	double dt, t, framet;					/*Simulation + frame times*/
	double beta,gamma;						/*Newmark-Beta Parameters*/
	int Bcase;								/*What boundary shape to take*/
	double H,HSQ, sr; 						/*Support Radius*/
} SIM;

typedef struct FLUID {
	double rho0; 						/*Resting density*/
	double Simmass, Boundmass;			/*Particle and boundary masses*/
	double alpha,eps,Cs, nu;	
	double sig, lam;					/*Fluid properties*/
	double gam, B; 
	double contangb;
}FLUID;

typedef struct Particle {
Particle(Vector2d x, Vector2d v, Vector2d f, double rho, double Rrho, double m, bool bound)	:
	xi(x), v(v), V(0.0,0.0),  f(f), Sf(0.0,0.0), rho(rho), p(0.0), Rrho(Rrho), m(m), c(0), b(bound) {}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Vector2d xi, v, V, f, Sf;
	double rho, p, Rrho, m, c;
	bool b;
	int size() const {return(xi.size());}
	double operator[](int a) const {
		return(xi[a]);
	}
}Particle;

typedef std::vector<std::vector<size_t>> outl;
typedef std::vector<Particle> State;
typedef KDTreeVectorOfVectorsAdaptor<State, double> my_kd_tree_t;
nanoflann::SearchParams params;

void write_header() 
{
	cout << "******************************************************************" << endl << endl;
	cout << "                              WCXSPH                              " << endl << endl;
	cout << "        Weakly Compressible Smoothed Particle Hydrodynamics       " << endl;
	cout << "                       with XSPH correction                       " << endl;
	cout << "                      for tide-breaking case                      " << endl << endl;
	cout << "                         James O. MacLeod                         " << endl;
	cout << "                    University of Bristol, U.K.                   " << endl << endl;
	cout << "******************************************************************" << endl << endl;
}

int getInt(ifstream& In)
{
	string line;
	getline(In,line);
	int i = stoi(line);
	return i;
}

double getDouble(ifstream& In)
{
	string line;
	getline(In,line);
	double d = stod(line);
	return d; 
}

void GetInput(int i,char* InFile, SIM *svar, FLUID *fvar)
{

	if (i>2)
	{
		cerr << "Too many inputs provided to the program." << endl;
		cerr <<" Only one input file please." << endl;
		exit(-1);
	}
	if (i==1)
	{
		cerr << "Too few arguments provided" << endl;
		cerr << "Will assume a standard set of parameters..." << endl;
		svar->xyPART(40,40); /*Number of particles in (x,y) directions*/
		svar->SimPts = svar->xyPART(0)*svar->xyPART(1); /*total sim particles*/
		svar->Pstep = 0.01;	/*Initial particle spacing*/
		svar->Bstep = 0.6; 	/*Boundary factor of particle spacing (dx = Pstep*Bstep)*/
		svar->Bcase = 1; 		/*Boundary case*/
		
		//Simulation window parameters
		svar->Box(10,5); 		/*Boundary dimensions*/
		svar->Start(0.2,0.2); /*Simulation particles start + end coords*/
		const static Vector2d Finish(svar->Start(0)+svar->Pstep*svar->xyPART(0)*1.0,
						svar->Start(1)+svar->Pstep*svar->xyPART(1)*1.0);

		// SPH Parameters
		svar->H =  3.0*svar->Pstep; /*Support Radius*/
		svar->HSQ = svar->H*svar->H; 
		svar->sr = 4*svar->HSQ; 	/*KDtree search radius*/

		//Timestep Parameters
		svar->dt = 0.0002;			/*Starting Timestep*/
		svar->t = 0.0;				/*Current time in sim*/
		svar->beta = 0.25;	/*Newmark-Beta parameter*/
		svar->gamma = 0.5;	
		svar->subits = 8;		/*Newmark-Beta iterations*/
		svar->Nframe = 2500;		/*Number of output frames*/

		//Fluid Properties
		fvar->rho0 = 1000.0;  /*Rest density*/
		fvar->Simmass = fvar->rho0*((Finish(0)-svar->Start(0))*
				(Finish(1)-svar->Start(1))/(1.0*svar->SimPts));
		fvar->Boundmass = 1.2*fvar->Simmass;
		fvar->Cs = 50; 	 			/*Speed of sound*/
		fvar->nu = 0.00000089;		/*Viscosity*/
		fvar->sig = 0.0728;			/*Surface tension*/
		fvar->lam = 6.0/81.0*pow((2.0*svar->H),4.0)/pow(M_PI,4.0)*
					(9.0/4.0*pow(M_PI,3.0)-6.0*M_PI-4.0);
		fvar->gam = 7.0;  							 /*Factor for Tait's Eq*/
		fvar->B = fvar->rho0*pow(fvar->Cs,2)/fvar->gam;  /*Factor for Tait's Eq*/
	}
		
	std::ifstream in(InFile);
  	if(in.is_open()) 
  	{	/*Simulation parameters*/
  		cout << "Input file opened. Reading settings..." << endl;
  		svar->framet = getDouble(in);
  		svar->Nframe = getInt(in);
  		svar->subits = getInt(in);
  		svar->beta = getDouble(in);
  		svar->gamma = getDouble(in);
  		svar->xyPART(0) = getInt(in); 
  		svar->xyPART(1) = getInt(in);
  		svar->SimPts = svar->xyPART(0)*svar->xyPART(1); /*total sim particles*/
  		svar->Start(0) = getDouble(in);
  		svar->Start(1) = getDouble(in);
  		svar->Bcase = getInt(in);
  		svar->Box(0) = getDouble(in);
  		svar->Box(1) = getDouble(in);
  		svar->Pstep = getDouble(in);
  		svar->Bstep = getDouble(in);
  		svar->H = getDouble(in); /*End of state read*/
  		svar->H*= svar->Pstep;
  		svar->HSQ = svar->H*svar->H;
  		svar->sr = 4*svar->HSQ;
  		svar->dt = 0.0002;
  		svar->t = 0.0;	

		/*Fluid parameters read*/
  		fvar->alpha = getDouble(in);
  		fvar->eps = getDouble(in);
  		fvar->contangb = getDouble(in);
  		fvar->rho0 = getDouble(in);
  		fvar->Cs = getDouble(in);
  		fvar->nu = getDouble(in);
  		fvar->sig = getDouble(in);

  		const static Vector2d Finish(svar->Start(0)+svar->Pstep*svar->xyPART(0)*1.0,
						svar->Start(1)+svar->Pstep*svar->xyPART(1)*1.0);
  		const static double Simmass = fvar->rho0*
	  		((Finish(0)-svar->Start(0))*(Finish(1)-svar->Start(1))/(1.0*svar->SimPts));
		fvar->Simmass = Simmass; /*Mass from spacing and density*/
		fvar->Boundmass = 1.2*Simmass;
		fvar->gam = 7.0;  							 /*Factor for Tait's Eq*/
		fvar->B = fvar->rho0*pow(fvar->Cs,2)/fvar->gam;  /*Factor for Tait's Eq*/
		fvar->lam = 6.0/81.0*pow((2.0*svar->H),4.0)/pow(M_PI,4.0)*
					(9.0/4.0*pow(M_PI,3.0)-6.0*M_PI-4.0);
  		in.close();
  		
  	}
  	else {
	    cerr << "Error opening the input file." << endl;
	    exit(-1);
  	}
}

///******Wendland's C2 Quintic Kernel*******
double W2Kernel(double dist,double H) 
{
	double q = dist/H;
	double correc = (7/(4*M_PI*H*H));
	return (pow(1-0.5*q,4))*(2*q+1)*correc;
}

/*Gradient*/
Vector2d W2GradK(Vector2d Rij, double dist,double H)
{
	double q = dist/H;
	double correc = (7/(4*M_PI*H*H));
	return 5.0*(Rij/(H*H))*pow(1-0.5*q,3)*correc;
}

/*2nd Gradient*/
double W2Grad2(Vector2d Rij, double dist,double H) 
{
	double q = dist/H;
	double correc = (7/(4*M_PI*H*H));
	return Rij.dot(Rij)*(5.0*correc/(H*H))*(2*q-1)*pow(1-0.5*q,2);
}

void FindNeighbours(State &part,my_kd_tree_t &mat_index, outl &outlist, SIM *svar)
{
	outlist.erase(outlist.begin(),outlist.end());
	double search_radius = svar->sr;
	/*Find neighbour list*/
	for(size_t i=0; i<svar->npts; ++i)
	{		
		std::vector<std::pair<size_t,double>> matches; /* Nearest Neighbour Search*/
		
		mat_index.index->radiusSearch(&part[i].xi[0], search_radius, matches, params);

		std::vector<size_t> temp;
		for (auto &j:matches) 
		{
			temp.emplace_back(j.first); 	
		}
		outlist.emplace_back(temp);		
	}		
}

void Forces(State &part, outl outlist,SIM *svar,FLUID *fvar) 
{
	maxmu=0; 				/* CFL Parameter */
	double alpha = fvar->alpha; 	/* Artificial Viscosity Parameter*/
	double eps = fvar->eps; 		/* XSPH Influence Parameter*/
	double numpartdens = 0.0;

	for (size_t i=0; i< svar->npts; ++i) 
	{
		for (size_t j=0; j<outlist[i].size(); ++j) 
		{ /* Surface Tension calcs */
			
			double r = (part[outlist[i][j]].xi-part[i].xi).norm();
			numpartdens += W2Kernel(r,svar->H);
		}	
	}
	numpartdens=numpartdens/(1.0*svar->npts);
	
	// #pragma omp parallel
	{
		// #pragma omp for
		
		for (size_t i=0; i< svar->bound_parts; ++i) 
		{	/*Find variation in density for the boundary (but don't bother with force)*/
			double Rrhocontr = 0.0;
			Particle pi = part[i];

			for(size_t j=0; j<outlist[i].size(); ++j)
			{
				Particle pj = part[outlist[i][j]];
				Vector2d Rij = pj.xi-pi.xi;
				Vector2d Vij = pj.v-pi.v;
				double r = Rij.norm();
				Vector2d Grad = W2GradK(Rij, r,svar->H);
				Rrhocontr -= pj.m*(Vij.dot(Grad));  
			}
			part[i].Rrho = Rrhocontr; /*drho/dt*/
		}

		for (size_t i=svar->bound_parts; i< svar->npts; ++i) 
		{	/*Do force calculation for fluid particles.*/
			Particle pi = part[i];
			pi.V = pi.v;

			double Rrhocontr = 0.0;
			Vector2d contrib= Vector2d::Zero();
			Vector2d visc = Vector2d::Zero();
			Vector2d SurfC= Vector2d::Zero(); 

			vector<double> mu;  /*Vector to find largest mu value for CFL stability*/
			mu.emplace_back(0);	/*Avoid dereference of empty vector*/
			
			for (size_t j=0; j < outlist[i].size(); ++j) 
			{	/*Find force and density variation for particles*/
				Particle pj = part[outlist[i][j]];

				/*Check if the position is the same, and skip the particle if yes*/
				if(pi.xi == pj.xi)
					continue;

				Vector2d Rij = pj.xi-pi.xi;
				Vector2d Vij = pj.v-pi.v;
				double r = Rij.norm();
				double Kern = W2Kernel(r,svar->H);
				Vector2d Grad = W2GradK(Rij, r,svar->H);
				
				/*Pressure and artificial viscosity calc - Monaghan 1994 p.400*/
				double rhoij = 0.5*(pi.rho+pj.rho);
				double cbar= 0.5*(sqrt((fvar->B*fvar->gam)/pi.rho)+sqrt((fvar->B*fvar->gam)/pj.rho));
				double vdotr = Vij.dot(Rij);
				double muij= svar->H*vdotr/(r*r+0.01*svar->HSQ);
				mu.emplace_back(muij);
				double pifac = alpha*cbar*muij/rhoij;
		
				if (vdotr > 0) pifac = 0;
				contrib += pj.m*Grad*(pifac - pi.p*pow(pi.rho,-2)-pj.p*pow(pj.rho,-2));

				/*Laminar Viscosity (Morris)*/
				visc -= Vij*(fvar->nu/pj.m)*((pi.rho+pj.rho)/(pow(0.5*(pi.rho+pj.rho),2)))
					*(1.0/(r*r+0.01*svar->HSQ))*Rij.dot(Grad);

				/*Surface Tension as described by Nair & Poeschel (2017)*/
				double fac=1;
				
				if(pj.b==true) 
		            fac=(1+0.5*cos(M_PI*(150/180)));
		        		        
				double sij = 0.5*pow(numpartdens,-2.0)*(fvar->sig/fvar->lam)*fac;
				SurfC -= (Rij/r)*sij*cos((3.0*M_PI*r)/(4.0*svar->H))/pj.m;

				/* XSPH Influence*/
				pi.V+=eps*(pj.m/rhoij)*Kern*Vij; 

				/*drho/dt*/
				Rrhocontr -= pj.m*(Vij.dot(Grad));	
			}

			pi.Rrho = Rrhocontr; /*drho/dt*/
			pi.f= contrib + SurfC;
			pi.Sf = SurfC;
			// pi.f(1) -= 9.81; /*Add gravity*/

			part[i]=pi; //Update the actual structure

			//CFL f_cv Calc
			double it = *max_element(mu.begin(),mu.end());
			if (it > maxmu)
				maxmu=it;
		} /*End of sim parts*/
	}
}

/*Density Reinitialisation using Least Moving Squares as in A. Colagrossi (2003)*/
void DensityReinit(State &p, outl outlist,SIM *svar)
{
	Vector3d one(1.0,0.0,0.0);
	for(size_t i=0; i< svar->npts; ++i)
	{
		Matrix3d A= Matrix3d::Zero();
		//Find matrix A.
		Particle pi = p[i];
		for (size_t j=0; j< outlist[i].size(); ++j) 
		{
			Particle pj = p[outlist[i][j]];
			Vector2d Rij = pi.xi-pj.xi;
			Matrix3d Abar;	
			Abar << 1      , Rij(0)        , Rij(1)        ,
				    Rij(0) , Rij(0)*Rij(0) , Rij(1)*Rij(0) ,
				    Rij(1) , Rij(1)*Rij(0) , Rij(1)*Rij(1) ;

			A+= W2Kernel(Rij.norm(),svar->H)*Abar*pj.m/pj.rho;
		}
				
		Vector3d Beta;
		//Check if A is invertible
		FullPivLU<Matrix3d> lu(A);
		if (lu.isInvertible())
			Beta = lu.inverse()*one;
		else
			Beta = (1/A(0,0))*one;

		//Find corrected kernel
		double rho = 0.0;
		for (size_t j=0; j< outlist[i].size(); ++j) 
		{
			Vector2d Rij = pi.xi-p[outlist[i][j]].xi;
			rho += p[outlist[i][j]].m*W2Kernel(Rij.norm(),svar->H)*
			(Beta(0)+Beta(1)*Rij(0)+Beta(2)*Rij(1));
		}

		p[i].rho = rho;
	}
}

void Newmark_Beta(State &pn, State &pnp1, my_kd_tree_t &mat_index,
	 outl &outlist,SIM *svar,FLUID *fvar) 
{
	vector<Vector2d> xih;
	xih.reserve(svar->npts);
	for (unsigned int k = 0; k < svar->subits; ++k)
	{	
		Forces(pnp1, outlist,svar,fvar); /*Guess force at time n+1*/

		/*Previous State for error calc*/
		for (size_t  i=0; i< svar->npts; ++i)
			xih[i] = pnp1[i].xi;

		/*Update the state at time n+1*/
		for (size_t i=0; i <svar->bound_parts; ++i) 
		{	/*Boundary Particles*/
			pnp1[i].rho = pn[i].rho+0.5*svar->dt*(pn[i].Rrho+pnp1[i].Rrho);
			pnp1[i].p = fvar->B*(pow(pnp1[i].rho/fvar->rho0,fvar->gam)-1);
		}
		for (size_t i=svar->bound_parts; i < svar->npts ; ++i )
		{	/*Fluid particles*/
			pnp1[i].v = pn[i].v+svar->dt*((1-svar->gamma)*pn[i].f+svar->gamma*pnp1[i].f);
			pnp1[i].rho = pn[i].rho+svar->dt*((1-svar->gamma)*pn[i].Rrho+svar->gamma*pnp1[i].Rrho);
			pnp1[i].xi = pn[i].xi+svar->dt*pn[i].V+0.5*(svar->dt*svar->dt)*(1-2*svar->beta)*pn[i].f
							+(svar->dt*svar->dt*svar->beta)*pnp1[i].f;
			pnp1[i].p = fvar->B*(pow(pnp1[i].rho/fvar->rho0,fvar->gam)-1);
		}
		
		mat_index.index->buildIndex();
		FindNeighbours(pnp1, mat_index, outlist,svar);

		errsum = 0.0;
		for (size_t i=0; i < pnp1.size(); ++i)
		{
			Vector2d r = pnp1[i].xi-xih[i];
			errsum += r.squaredNorm();
		}

		if(k == 0)
			logbase=log10(sqrt(errsum/(1.0*svar->npts)));
	}

	
	/*Find maximum safe timestep*/
	vector<Particle>::iterator maxfi = std::max_element(pnp1.begin(),pnp1.end(),
		[](Particle p1, Particle p2){return p1.f.norm()< p2.f.norm();});
	double maxf = maxfi->f.norm();
	double dtf = sqrt(svar->H/maxf);
	double dtcv = svar->H/(fvar->Cs+maxmu);
	svar->dt = 0.5*min(dtf,dtcv);

	//Update the state at time n
	pn = pnp1;
}


void InitSPH(State &particles, SIM *svar, FLUID *fvar) 
{
	cout << "Initialising simulation with " << svar->SimPts << " particles" << endl;
	
	//Structure input initialiser
	//Particle(Vector2d x, Vector2d v, Vector2d vh, Vector2d f, float rho, float Rrho, bool bound) :
	Vector2d v = Vector2d::Zero();  
	Vector2d f = Vector2d::Zero(); 
	double rho=fvar->rho0; 
	double Rrho=0.0; 
	 
	/*create the boundary particles*/ 	 
	static double stepx = svar->Pstep*svar->Bstep;
	static double stepy = svar->Pstep*svar->Bstep;
	const static int Ny = ceil(svar->Box(1)/stepy);
	stepy = svar->Box(1)/Ny;
	const static int Nx = ceil(svar->Box(0)/stepx);
	stepx = svar->Box(0)/Nx;
	switch (svar->Bcase) 
	{
		case 0:
		{ /*No boundary*/
			break;
		}
		case 1: /*Rectangle*/
		{
			for(int i = 0; i <= Ny ; ++i) {
				Vector2d xi(0.f,i*stepy);
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Boundmass,true));
			}
			// for(int i = 1; i <Nx ; ++i) {
			// 	Vector2d xi(i*stepx,Box(1));
			// 	particles.emplace_back(Particle(xi,v,f,rho,Rrho,Boundmass,Bound));	
			// }
			Vector2d x(stepx,(Ny+0.5)*stepy);
			particles.emplace_back(Particle(x,v,f,rho,Rrho,fvar->Boundmass,true));
			x(0) = svar->Box(0) -stepx;
			particles.emplace_back(Particle(x,v,f,rho,Rrho,fvar->Boundmass,true));

			for(int i= Ny; i>0; --i) {
				Vector2d xi(svar->Box(0),i*stepy);
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Boundmass,true));	
			}
			for(int i = Nx; i > 0; --i) {
				Vector2d xi(i*stepx,0.f);
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Boundmass,true));
			}
			break;
		}
		case 2: /*Bowl*/
		{
			
			double r= svar->Box(0);
			double dtheta = atan((svar->Pstep*svar->Bstep)/r);
			for (double theta = 0.0; theta < M_PI; theta+=dtheta)
			{
				Vector2d xi(-r*cos(theta),r*(1-sin(theta)));
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Boundmass,true));
			}
			break;
		}
		default: 
		{
			break;
		}
	}
	
	svar->bound_parts = particles.size();
	
	/*Create the simulation particles*/
	for( int i=0; i< svar->xyPART(0); ++i) 
	{
		for(int j=0; j< svar->xyPART(1); ++j)
		{				
				Vector2d xi(svar->Start(0)+i*svar->Pstep,svar->Start(1)+j*svar->Pstep);		
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Simmass,false));
		}
	}

	/*Create a second droplet to hit the first*/
	v(0) = -3.0;
	for( int i=0; i< 10; ++i) 
	{
		for(int j=0; j< 10; ++j)
		{				
				Vector2d xi(1.0+i*svar->Pstep,0.37+j*svar->Pstep);		
				particles.emplace_back(Particle(xi,v,f,rho,Rrho,fvar->Simmass,false));
		}
	}


	svar->npts = particles.size();
	svar->SimPts = svar->npts;
	if(svar->npts!=svar->bound_parts+svar->SimPts)
	{
		cerr<< "Mismatch of particle count." << endl;
		cerr<< "Particle array size doesn't match defined values." << endl;
		exit(-1);
	}
	cout << "Total Particles: " << svar->npts << endl;
}

void write_settings(SIM *svar, FLUID *fvar)
{
	std::ofstream fp("Test_Settings.txt", std::ios::out);

  if(fp.is_open()) {
    //fp << "VERSION: " << VERSION_TAG << std::endl << std::endl; //Write version
    fp << "SIMULATION PARAMETERS:" << std::endl; //State the system parameters.
    fp << "\tNumber of frames (" << svar->Nframe <<")" << std::endl;
    fp << "\tParticle Spacing ("<< svar->Pstep << ")" << std::endl;
    fp << "\tParticle Mass ("<< fvar->Simmass << ")" << std::endl;
    fp << "\tReference density ("<< fvar->rho0 << ")" << std::endl;
    fp << "\tSupport Radius ("<< svar->H << ")" << std::endl;
    fp << "\tGravitational strength ("<< 9.81 << ")" << std::endl;
    fp << "\tNumber of boundary points (" << svar->bound_parts << ")" << std::endl;
    fp << "\tNumber of simulation points (" << svar->SimPts << ")" << std::endl;
    fp << "\tIntegrator type (Newmark_Beta)" << std::endl;

    fp.close();
  }
  else {
    cerr << "Error opening the settings output file." << endl;
    exit(-1);
  }
}

void write_frame_data(State particles, std::ofstream& fp, double t,SIM *svar)
{	
	switch(svar->Bcase) 
	{
		case 0:
		{ /*In the case of no boundary, don't write anything*/
		  	break;
		}
		default:
		{
			fp <<  "ZONE T=\"Boundary Data\"" << ", I=" << svar->bound_parts << ", F=POINT" <<
		    ", STRANDID=1, SOLUTIONTIME=" << t << std::endl;
		  	for (auto b=particles.begin(); b!=std::next(particles.begin(),svar->bound_parts); ++b)
			{
		        fp << b->xi(0) << " " << b->xi(1) << " ";
		        fp << b->v.norm() << " ";
		        fp << b->f.norm() << " ";
		        fp << b->rho << " "  << b->p  << " " << b->Sf.norm() << std::endl;
		  	}
			break;
		}
	}
    
    fp <<  "ZONE T=\"Particle Data\"" <<", I=" << svar->SimPts << ", F=POINT" <<
    ", STRANDID=2, SOLUTIONTIME=" << t  << std::endl;
    unsigned int i=0;
  	for (auto p=std::next(particles.begin(),svar->bound_parts); p!=particles.end(); ++p)
	{
		/*if (p->xi!=p->xi || p->v!=p->v || p->f!=p->f) {
			cerr << endl << "Simulation is broken. A value is nan." << endl;
			cerr << "Broken line..." << endl;
			cerr << p->xi(0) << " " << p->xi(1) << " ";
	        cerr << p->v.norm() << " ";
	        cerr << p->f.norm() << " ";
	        cerr << p->rho << " " << p->p << std::endl; 
	        fp.close();
			exit(-1);
		}*/
        fp << p->xi(0) << " " << p->xi(1) << " ";
        fp << p->v.norm() << " ";
        fp << p->f.norm() << " ";
        fp << p->rho << " "  << p->p 
        << " " << p->Sf.norm() /* << " " << nstore[i].norm()*/ << std::endl; 
        ++i;
  	}
}

int main(int argc, char *argv[]) 
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	high_resolution_clock::time_point t2;
    double duration;

    write_header();

    static SIM simvar;
    static FLUID fluidvar;
    GetInput(argc, argv[1], &simvar,&fluidvar);

 	
	//initialise the std::vector<Particle> particles memory
	State particles;	/*Particles at n*/
	State particlesh; 	/*Particles at n+1*/
	InitSPH(particles,&simvar,&fluidvar);

	for (auto p: particles)
			particlesh.emplace_back(p);
	
	/*Tree algorithm stuff*/
	my_kd_tree_t mat_index(2,particlesh,10);
	mat_index.index->buildIndex();
	outl outlist;
	outlist.reserve(simvar.npts);
	FindNeighbours(particlesh, mat_index, outlist,&simvar);

	Forces(particlesh,outlist,&simvar,&fluidvar); /*Perform an iteration to populate the vectors*/

	write_settings(&simvar,&fluidvar);

	/*Open simulation files*/
	std::ofstream f1("Test.plt", std::ios::out);
	std::ofstream f2("frame.info", std::ios::out);
	if (f1.is_open())
	{
		f1 << std::fixed << setprecision(6);
		f2 << std::fixed << setprecision(4);
		cout << std::fixed << setprecision(4);
		//Write file header defining variable names
		f1 << "TITLE = \"WCXSPH Output\"" << std::endl;
		f1 << "VARIABLES = \"x (m)\", \"y (m)\", \"v (m/s)\", \"a (m/s<sup>-1</sup>)\", " << 
			"\"<greek>r</greek> (kg/m<sup>-3</sup>)\", \"P (Pa)\", \"SurfC\"" << std::endl;
		write_frame_data(particlesh,f1,0,&simvar);

		t2 = high_resolution_clock::now();
		duration = duration_cast<microseconds>(t2-t1).count()/1e6;
		cout << "Frame: " << 0 << "  Sim Time: " << simvar.t << "  Compute Time: " 
		<< duration <<"  Error: " << log10(sqrt(errsum/(1.0*simvar.npts)))-logbase << endl;
		f2 << "Frame: " << 0 << "  S Time: " << simvar.t << "  C Time: " 
			<< duration << "  Error: " << log10(sqrt(errsum/(1.0*simvar.npts)))-logbase << 
			" Its: " << 0 << endl; 

		const static int outframe = 20;
		for (unsigned int frame = 1; frame<= simvar.Nframe; ++frame) 
		{	
			int stepits=0;	
			double stept=0.0;		  
			while (stept<0.002) 
			{
				
			    Newmark_Beta(particles,particlesh,mat_index,outlist,&simvar,&fluidvar);
			    simvar.t+=simvar.dt;
			    stept+=simvar.dt;
			    ++stepits;
			}
			

			t2= high_resolution_clock::now();
			duration = duration_cast<microseconds>(t2-t1).count()/1e6;
			f2 << "Frame: " << frame << "  S-Time: " << simvar.t << "  C-Time: " 
			<< duration <<"  Error: " << log10(sqrt(errsum/(1.0*simvar.npts)))-logbase << 
			" Its: " << stepits << endl;  

			if (frame % outframe == 0 )
			{
			  	cout << "Frame: " << frame << "  Sim Time: " << simvar.t-simvar.dt << "  Compute Time: " 
			  	<< duration <<"  Error: " << log10(sqrt(errsum/(1.0*simvar.npts)))-logbase << endl;
			}

			DensityReinit(particlesh,outlist,&simvar);
			write_frame_data(particlesh,f1,simvar.t-simvar.dt,&simvar);
		}
		f1.close();
		f2.close();
	}
	else
	{
		cerr << "Error opening frame output file." << endl;
		exit(-1);
	}

	cout << "Simulation complete!" << endl;
    
    cout << "Time taken:\t" << duration << " seconds" << endl;
    cout << "Total simulation time:\t" << simvar.t << " seconds" << endl;
	return 0;
}