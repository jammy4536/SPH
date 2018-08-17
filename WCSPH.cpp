/*** WCSPH (Weakly Compressible Smoothed Particle Hydrodynamics) Code***/
/********* Created by Jamie MacLeod, University of Bristol *************/
/*** Force Calculation: On Simulating Free Surface Flows using SPH. Monaghan, J.J. (1994) ***/
/*** Smoothing Kernel: Wendland's C2 ***/
/*** Integrator: Newmark-Beta ****/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <float.h>
#include <string.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <dirent.h>
#include <chrono>
#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <utils.h>
#include <KDTreeVectorOfVectorsAdaptor.h>
#include <cstdlib>


#ifndef M_PI
#define M_PI (4.0*atan(1.0))
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace nanoflann;

/*Make these from input file at some point...*/
//Simulation Definition
const static Vector2i xyPART(25,38); /*Number of particles in (x,y) directions*/
const static unsigned int SimPts = xyPART(0)*xyPART(1); /*total sim particles*/
static size_t bound_parts;			/*Number of boundary particles*/
const static double Pstep = 0.2;	/*Initial particle spacing*/

//Simulation window parameters
const static double BoxW = 15.0; /*Boundary width*/
const static double BoxH = 8.0; /*Boundary height*/
const static Vector2d Start(0.31,0.31); /*Simulation particles start + end coords*/
const static Vector2d Finish(Start(0)+Pstep*xyPART(0)*1.0,Start(1)+Pstep*xyPART(1)*1.0);

//Fluid Properties
const static double rho0 = 1000.0; /*Rest density*/
const static double mass = rho0*((Finish(0)-Start(0))*(Finish(1)-Start(1))/(1.0*SimPts));
const static double Cs = 50; /*Speed of sound*/
const static double gam = 7.0; /*Factor for Tait's Eq*/
const static double B = rho0*pow(Cs,2)/gam; /*Factor for Tait's Eq*/

// SPH Parameters
const static double H = sqrt((3/M_PI)*(mass/rho0)); /*Support Radius*/
const static double HSQ = H*H;
const static double r0 = Pstep;		/*Boundary support radius*/
const static double D = pow(Cs,2);	/*Boundary param 1*/
const static float N1 = 4;			/*Boundary param 2*/
const static float N2 = 2;			/*Boundary param 3*/
const static double dt = 0.0002;	/*Timestep*/
static float t = 0.f;				/*Current time in sim*/
const static float beta = 0.25;		/*Newmark-Beta parameter*/
const static int subits = 8;		/*Newmark-Beta iterations*/
const static int Nframe = 320;		/*Number of output frames*/
const static int Ndtframe = 60; 	/*Timesteps per frame*/
const static Vector2d zero(0.0,0.0);


typedef struct Particle {
Particle(Vector2d x, Vector2d v, Vector2d f, float rho, float Rrho, bool bound)	:
	xi(x), v(v),  f(f), rho(rho), p(0.0), Rrho(Rrho), b(bound),
	fp(0.0,0.0), fb(0.0,0.0), fv(0.0,0.0), fs(0.0,0.0) {}
	Vector2d xi, v, f;
	float rho, p, Rrho;
	bool b;
	Vector2d fp, fb, fv, fs;
	int size() const {return(xi.size());}
	double operator[](int a) const {
		return(xi[a]);
	}
}Particle;

typedef std::vector<Particle> State;
typedef KDTreeVectorOfVectorsAdaptor<State, double> my_kd_tree_t;
const static double search_radius = 4*HSQ+0.001*H;
nanoflann::SearchParams params;

/*Smoothing Kernel*/
double Kernel(double dist)
{
	double q = dist/H;
	if (q<3)
		return exp(-q*q)/(M_PI*HSQ);
	else
		return 0;
}

/*Smoothing Kernel Gradient*/
Vector2d GradK(Vector2d Rij, double dist)
{
	double q = dist/H;
	if (q<3)
		return 2*(Rij/HSQ)*exp(-q*q)/(M_PI*HSQ);
	else
		return Vector2d(0,0);
}

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
	double alpha = 0.025;
	for (auto &pi :part)
	{

		//Reset the particle to zero
		pi.f = zero;
		pi.fp = zero;
		pi.fb = zero;
		pi.fv = zero;
		Vector2d contrib(0.0,0.0);
		double Rrhocontr = 0.0;
		pi.Rrho=0.0;
		std::vector<std::pair<size_t,double>> matches;
		mat_index.index->radiusSearch(&pi.xi[0], search_radius, matches, params);
		for (auto &i: matches)
		{
			Particle pj = part[i.first];
			Vector2d Rij = pj.xi-pi.xi;
			Vector2d Vij = pj.v-pi.v;
			double r = Rij.norm();
			Vector2d Grad = W2GradK(Rij, r);

			// if (pj.b==false) {
			/*Pressure and artificial viscosity calc - Monaghan 1994 p.400*/
				double cbar= 0.5*(sqrt((B*gam)/pi.rho)+sqrt((B*gam)/pj.rho));
				double vdotr = Vij(0)*Rij(0)+Vij(1)*Rij(1);
				double rhoij = 0.5*(pi.rho+pj.rho);
				double muij= H*vdotr/(r*r+0.01*HSQ);
				double pifac = alpha*cbar*muij/rhoij;

				if (vdotr > 0) pifac = 0;

				contrib += Grad*(pifac + pi.p/pow(pi.rho,2)+ pj.p/pow(pj.rho,2));

			//}

			// if (pj.b == true && r < r0)
			// {
			// 	contrib -= D*(pow((r0/r),N1)-pow((r0/r),N2))*Rij/Rij.squaredNorm();
			// 	pi.fb+=contrib*mass;
			// }


			Rrhocontr += (Vij.dot(Grad));


		}
		pi.Rrho = Rrhocontr*mass;
		pi.f= contrib*mass;
		pi.f(1) += -9.81; /*Add gravity*/
		pi.fp = contrib*mass;

		//cout << pi.f << endl;
	}

}

void PredictorCorrector(State &p, State &ph, my_kd_tree_t &mat_index)
{

	/*Predict*/
	Forces(p, mat_index); /*Find forces at time n*/

	// for (auto i=0; i< p.size(); ++i)
	// 	cout << p[i].f(0) << "\t" << p[i].f(1) << endl;

	/*Set boundary forces to zero to stop movement of the boundary*/
	for (size_t i=0; i < bound_parts; ++i)
		p[i].f = zero;

	for (size_t i=0; i < p.size() ; ++i )
	{
		ph[i].v = p[i].v+0.5*dt*p[i].f;
		ph[i].rho = p[i].rho+0.5*dt*p[i].Rrho;
		ph[i].xi = p[i].xi+0.5*dt*p[i].v;
		ph[i].p = B*(pow(ph[i].rho/rho0,gam)-1);
	}

	/*Correct*/
	Forces(ph, mat_index); /*Find forces at time n+1/2 */
	for (size_t i=0; i < bound_parts; ++i)
		ph[i].f = zero;

	for (size_t i=0; i < p.size() ; ++i )
	{
		ph[i].v = p[i].v+0.5*dt*ph[i].f;
		ph[i].rho = p[i].rho+0.5*dt*ph[i].Rrho;
		ph[i].xi = p[i].xi+0.5*dt*ph[i].v;
	}

	/*Update*/
	for (size_t i=0; i < p.size() ; ++i )
	{
		p[i].v = 2*ph[i].v - p[i].v;
		p[i].rho = 2*ph[i].rho - p[i].rho;
		p[i].xi = 2*ph[i].xi - p[i].xi;
		p[i].p = B*(pow(p[i].rho/rho0,gam)-1);
	}
	t+=dt;
}

void Newmark_Beta(State &pn, State &pnp1, my_kd_tree_t &mat_index)
{
	for (int k = 0; k < subits; ++k)
	{
		Forces(pnp1, mat_index); /*Guess force at time n+1*/
		for (size_t i=0; i < bound_parts; ++i)
		{
			pnp1[i].f = zero; /*Zero boundary forces*/
			pn[i].f = zero;   /*So that they dont move*/
		}

		/*Update the state at time n+1*/
		for (size_t i=0; i < pn.size() ; ++i )
		{
			pnp1[i].v = pn[i].v+0.5*dt*(pn[i].f+pnp1[i].f);
			pnp1[i].rho = pn[i].rho+0.5*dt*(pn[i].Rrho+pnp1[i].Rrho);
			pnp1[i].xi = pn[i].xi+dt*pn[i].v+0.5*(dt*dt)*(1-2*beta)*pn[i].f+(dt*dt*beta)*pnp1[i].f;
			pnp1[i].p = B*(pow(pnp1[i].rho/rho0,gam)-1);
		}
		mat_index.index->buildIndex();
	}

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

	static double stepx = r0*0.7;
	static double stepy = r0*0.7;
	const static int Ny = ceil(BoxH/stepy);
	const static int Nx = ceil(BoxW/stepx);
	stepx = BoxW/Nx;
	stepy = BoxH/Ny;
	for(int i = 0; i <= Ny ; ++i) {
		Vector2d xi(0.f,i*stepy);
		particles.push_back(Particle(xi,v,f,rho,Rrho,true));
	}
	for(int i = 1; i <Nx ; ++i) {
		Vector2d xi(i*stepx,BoxH);
		particles.push_back(Particle(xi,v,f,rho,Rrho,true));
	}
	for(int i= Ny; i>0; --i) {
		Vector2d xi(BoxW,i*stepy);
		particles.push_back(Particle(xi,v,f,rho,Rrho,true));
	}
	for(int i = Nx; i > 0; --i) {
		Vector2d xi(i*stepx,0.f);
		particles.push_back(Particle(xi,v,f,rho,Rrho,true));
	}

	bound_parts = particles.size();


	/*Create the simulation particles*/
	for( int i=0; i< xyPART(0); ++i)
	{
		for(int j=0; j< xyPART(1); ++j)
		{
				Vector2d xi(Start(0)+i*Pstep,Start(1)+j*Pstep);
				particles.push_back(Particle(xi,v,f,rho,Rrho,false));
		}
	}
	cout << "Total Particles: " << SimPts + bound_parts << endl;
}
void write_settings()
{
	std::ofstream fp("Test_Settings.txt", std::ios::out);
  //std::ofstream fp("Test_Settings.txt", std::ios::out);
  if(fp.is_open()) {
    //fp << "VERSION: " << VERSION_TAG << std::endl << std::endl; //Write version
    fp << "SIMULATION PARAMETERS:" << std::endl; //State the system parameters.
    fp << "\tNumber of frames (" << Nframe <<")" << std::endl;
    fp << "\tSteps per frame ("<< Ndtframe << ")" << std::endl;
    fp << "\tTime step (" << dt << ")" << std::endl;
    fp << "\tParticle Spacing ("<< Pstep << ")" << std::endl;
    fp << "\tParticle Mass ("<< mass << ")" << std::endl;
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
		//Vector2d a=  p.f/p.rho;
        fp << b->xi(0) << " " << b->xi(1) << " ";
        fp << b->v(0) << " " << b->v(1) << " ";
        fp << b->f(0) << " " << b->f(1) << " ";
        fp << b->rho << " "  << b->p << std::endl;
  	}

    fp <<  "ZONE t=\"Particle Data\", STRANDID=1, SOLUTIONTIME=" << t << ", DATAPACKING=POINT" << std::endl;
  	for (auto p=std::next(particles.begin(),bound_parts); p!=particles.end(); ++p)
	{
		//Eigen::Vector2d a=  p->f/p->rho;
		if (p->xi!=p->xi || p->v!=p->v || p->f!=p->f) {
			cerr << endl << "Simulation is broken. A value is nan." << endl;
			cerr << "Broken line..." << endl;
			cerr << p->xi(0) << " " << p->xi(1) << " ";
	        cerr << p->v(0) << " " << p->v(1) << " ";
	        cerr << p->f(0) << " " << p->f(1) << " ";
	        cerr << p->rho << " " << p->p << std::endl;
	        fp.close();
			exit(-1);
		}
        fp << p->xi(0) << " " << p->xi(1) << " ";
        fp << p->v(0) << " " << p->v(1) << " ";
        fp << p->f(0) << " " << p->f(1) << " ";
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
	for (auto p: particles)
			particlesh.push_back(Particle(p.xi,p.v,p.f,p.rho,p.Rrho,p.b));

	my_kd_tree_t mat_index(2,particlesh,10);
	mat_index.index->buildIndex();

	Forces(particles, mat_index);
	for (size_t i=0; i < bound_parts; ++i)
		particles[i].f = zero;

	/*Write settings*/
	write_settings();

	/*Open simulation file*/
	std::ofstream f1("Test.plt", std::ios::out);

	if (f1.is_open())
	{
		f1 << std::fixed << setprecision(3);

		cout << "Starting simulation..." << endl;
		//Write file header defining veriable names
		f1 <<  "VARIABLES = x, y, Vx, Vy, fx, fy, rho, P" << std::endl;
		write_frame_data(particles, f1);

		for (int frame = 1; frame<= Nframe; ++frame) {
				  cout << "Frame Number: " << frame << endl;
				  for (int i=0; i< Ndtframe; ++i) {
				    Newmark_Beta(particles, particlesh, mat_index);
				  }
				   write_frame_data(particles, f1);
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
