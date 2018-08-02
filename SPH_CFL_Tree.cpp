/*Smoothed Particle Hydrodynamics (SPH) Generator */
/* Test Code for single thread (Functional)*/

#include <iostream>
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
#define M_PI (3.14159265358979323846)
#endif

using namespace std;
using namespace std::chrono;
//using namespace Eigen;
using namespace nanoflann;
#define VERSION_TAG "SPHView01"

//Solver Parameters
/*Fluid Properties*/
const static Eigen::Vector2d G(0.f, 10*-9.81f); /*External forces*/
const static float REST_DENS = 1000.f;   /*Resting density*/
const static float GAS_CONST = 1000.f;   /*Gas constant for Eq Of State*/
const static float GAMMA = 7.f; 		 /*Gas constant for Eq Of State*/
const static float K = 1000; 			 /*Stiffness constant*/
const static float C_s = 50; 		 /*Gas constant for Eq Of State*/
const static float B = REST_DENS*pow(C_s,2.f)/GAMMA ; /*Gas constant for Eq Of State*/
const static float KAPPA = 0.01f; 		 /*Surface Tension*/

/*SPH Parameters*/
const static double H= 10.f; 	   /* Kernel Radius */
const static double HSQ = H*H; 	   /*Kernel squared*/
const static double D = pow(C_s,2); /*Boundary influnce param 1*/
const static double r0 = H/1.f; 	   /*Boundary influence cutoff radius*/
const static float N1 = 4.f; 	   /*Boundary influnce param 1*/
const static float N2 = 2.f; 	   /*Boundary influnce param 2*/
const static double MASS = 65.f;    /*Mass (assuming constant between particles)*/
const static double VISC = 60.f;   /*Viscosity Constant*/
const static double DT = 0.0008f;   /*Timestep*/
const static int NFRAME = 1000;	   /*Number of output frames*/
const static int NPFRAME = 30;	   /*Timesteps per frame*/
static size_t bound_parts;			   /*Number of boundary particles*/
const static char integrator = 'L';

/*TIMESTEP CALCULATION*/
const static float ALPHA_v= 0.4f;
const static float ALPHA_f=0.25f;
//static float DT = 0.001f;
static float t = 0.f;

//Smoothing kernels defined by Mueller.
const static double POLY6 = 315.f/(65.f*M_PI*pow(H,9.f));
const static double SPIKY_GRAD = -45.f/(M_PI*pow(H,6.f));
const static double VISC_LAP = 45.f/(M_PI*pow(H,6.f));

//Particle Data structure
typedef struct Particle {
	Particle(double _x, double _y, bool bound) : 
	x(_x,_y), v(0.f,0.f), vh(0.f,0.f), f(0.f,0.f), rho(0), p(0.f), b(bound){}
	Eigen::Vector2d x;
	Eigen::Vector2d v, vh, f;
	float rho, p;
	bool b;
	int size() const {return(x.size());}
	double operator[](int a) const {
		return(x[a]);
	}
}Particle;

//static vector<Particle> particles;
typedef vector<Particle> MyVector;
typedef KDTreeVectorOfVectorsAdaptor<MyVector, double> my_kd_tree_t;
const static double search_radius = HSQ+0.1f;
nanoflann::SearchParams params;


//Interaction
//const static int MAX_PARTICLES = 2500;
const static int DAM_PARTICLES = 2000;



//Simulation window parameters
const static double VIEW_WIDTH = 1200.f;
const static double VIEW_HEIGHT = 900.f;

vector<pair<size_t,double>> find_neighbors(Particle pi, my_kd_tree_t &mat_index)
{
	std::vector<std::pair<size_t,double>> matches;
	mat_index.index->radiusSearch(&pi.x[0], search_radius, matches, params);
	return matches;
}



/*** EULER FORWARD INTEGRATION***/
void Integrate (MyVector &particles, my_kd_tree_t &mat_index) 
{
	for (auto p=std::next(particles.begin(),bound_parts+1); p!=particles.end(); ++p)
	{
		// forward Euler integration
		p->v += DT*p->f/p->rho;
		p->x += DT*p->v;
	}

	mat_index.index->buildIndex();
}

/*** LEAPFROG TIME-STEP INTEGRATION ***/
// The velocity is updated at half-steps, and positions at full-steps.
// To get v^(i+1), compute another half-step from v^(i+1/2), using a^i
// At t=0, there's no vh, so perform a half step to get vh[0].
void leapfrog_start(MyVector &particles, my_kd_tree_t &mat_index)
{
	for (auto p=std::next(particles.begin(),bound_parts+1); p!=particles.end(); ++p)
	{
			p->vh = p->v + (p->f/p->rho)*(DT/2.f);
			p->v += DT*p->f/p->rho;
			p->x += p->vh*DT;	
	}
	mat_index.index->buildIndex();
}

void leapfrog_step(MyVector &particles, my_kd_tree_t &mat_index)
{
   for (auto p=std::next(particles.begin(),bound_parts+1); p!=particles.end(); ++p) 
   {
			p->vh += (p->f/p->rho)*(DT);
			p->v   = p->vh + (DT/2.f)*p->f/p->rho;
			p->x  += p->vh*DT;
	}
	mat_index.index->buildIndex();
}


//Compute the Density and Pressure for each particle
void ComputeDensityPressure (MyVector &particles, my_kd_tree_t &mat_index) 
{ 
	for (auto pi=particles.begin(); pi!=particles.end(); ++pi) 
	{
		pi->rho=0.f;


		std::vector<std::pair<size_t,double>> matches;
		matches= find_neighbors(*pi,mat_index);
		for (auto i : matches)
		{
			Eigen::Vector2d rij = particles[i.first].x-pi->x;
			float r2 = rij.squaredNorm();

			if(r2<HSQ) 
			{
				pi->rho +=MASS*POLY6*pow(HSQ-r2,3.f);
			}
		}
		/*Ideal Gas Law*/
		//pi.p = GAS_CONST*(pi.rho-REST_DENS);

		/*Tait's Equation*/
		pi->p = B*(pow((pi->rho/REST_DENS),GAMMA)-1);
	}
}

//Compute the forces on each particle in vector form
void ComputeForces(MyVector &particles, my_kd_tree_t &mat_index) 
{
	nanoflann::SearchParams params;
	params.sorted = true;
	//static float V_MAX=0.f;
	//static float F_MAX=0.f;
	for (auto pi=std::next(particles.begin(),bound_parts); pi!=particles.end(); pi++) 
	{
		Eigen::Vector2d fpress(0.f,0.f);
		Eigen::Vector2d fvisc(0.f,0.f);
		Eigen::Vector2d fsurft(0.f,0.f);
		Eigen::Vector2d fbound(0.f,0.f);
		
		std::vector<std::pair<size_t,double>> matches;
		matches = find_neighbors(*pi, mat_index);
		for (auto i : matches)
		{
			Particle pj = particles[i.first];
			
			if (&pi->x==&pj.x)
				continue; 

			Eigen::Vector2d rij = pi->x - pj.x;
			float r = rij.norm();
			if (pj.b ==false) 
			{
				/*Compute pressure contribution*/
				fpress += rij.normalized()*MASS*(pi->p + pj.p)/
					(2.f*pj.rho) * SPIKY_GRAD*pow(H-r,2.f);
				
				/*Compute Viscosity contribution*/
				fvisc += VISC*MASS*(pj.v - pi->v)/pj.rho *VISC_LAP*(H-r);
				
				/*Compute Surface tension contribution*/
				 // float r2 = rij.squaredNorm();
				 // fsurft -= pi->rho*KAPPA*POLY6*pow(HSQ-r2,3.f)*rij;
			}
			
			if (pj.b==true)
			{ /*Boundary effects */
				fbound += D*(pow((r0/r),N1)-pow((r0/r),N2))*rij/rij.squaredNorm();
			}
			
		}

		/*       Gravity forces           */
		Eigen::Vector2d fgrav = G*pi->rho;
		pi->f = fpress + fvisc + fgrav + fbound + fsurft;
		
		// if (pi->v.norm()> V_MAX)
		// 	V_MAX = pi->v.norm();
		// if (pi->f.norm() > F_MAX)
		// 	F_MAX = pi->f.norm();
	}

	//DT = min(min(0.001f,ALPHA_f*H/V_MAX), ALPHA_f*sqrt(H/F_MAX));
	
}

//Initialise the simulation, and run a timestep to obtain the particle data.
void InitSPH(MyVector &particles) 
{
	cout << "Initialising simulation with " << DAM_PARTICLES << " particles" << endl;
	
	/*create the boundary particles*/ 
	const static double step = r0*0.7f;
	for(double y = 0.f; y <= VIEW_HEIGHT ; y+=step) 
		particles.push_back(Particle(0.f,y,true));
	for(double x = step; x <VIEW_WIDTH ; x+=step) 
		particles.push_back(Particle(x,VIEW_HEIGHT,true));
	for(double y = VIEW_HEIGHT; y > 0.f; y-=step) 
		particles.push_back(Particle(VIEW_WIDTH,y,true));
	for(double x = VIEW_WIDTH; x > 0.f; x-=step)	
		particles.push_back(Particle(x,0.f,true));
	
	bound_parts= particles.size();
	cout << bound_parts << endl;

	/*Create the simulation particles*/
	for( double y = H*4.f; y < VIEW_HEIGHT - H*2.f; y+=H) 
	{
		for(double x = VIEW_WIDTH/4; x<= VIEW_WIDTH/2; x+=H)
		{
			if (particles.size()-bound_parts <= DAM_PARTICLES) 
			{
				//float jitter = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				particles.push_back(Particle(x/*+jitter*/,y,false));
			}
		}
	}	
}

void Step_1(MyVector &particles, my_kd_tree_t &mat_index) {
	ComputeDensityPressure(particles, mat_index);
	ComputeForces(particles, mat_index);
	
	switch (integrator)
	{
		case 'E':
			Integrate(particles, mat_index);
		case 'L':
			leapfrog_start(particles, mat_index);
	}
	t+=DT;
}

void Update_leapfrog(MyVector &particles, my_kd_tree_t &mat_index) 
{
	
	ComputeDensityPressure(particles, mat_index);
	ComputeForces(particles, mat_index);
	leapfrog_step(particles, mat_index);
	t+=DT;
}

void Update_Euler(MyVector &particles, my_kd_tree_t &mat_index)
{
	
	ComputeDensityPressure(particles, mat_index);
	ComputeForces(particles, mat_index);
	Integrate(particles, mat_index);
	t+=DT;
}

static void write_settings(char* cwd)
{
  DIR *theFolder=opendir(cwd);

  /***** DELETE ALL FILES IN THE OUTPUT FOLDER TO DO WITH PREVIOUS SIMULATION *****/
  struct dirent *next_file;
  char filepath[256];
  while ((next_file = readdir(theFolder))!=NULL) {
    sprintf(filepath, "%s/%s",cwd,next_file->d_name);
    remove(filepath);
  }
  closedir(theFolder);
  /**************************/
  //Add in the output folder to the path.
  string str = cwd;
  str.append("Test");
  str.append("_settings.txt");
  //insert a settings file to show the simulation settings.
  //cout << str << endl;
  std::ofstream fp(str.c_str(), std::ios::out);
  //std::ofstream fp("Test_Settings.txt", std::ios::out);
  if(fp.is_open()) {
    fp << "VERSION: " << VERSION_TAG << std::endl << std::endl; //Write version
    fp << "SIMULATION PARAMETERS:" << std::endl; //State the system parameters.
    fp << "\tNumber of frames (" << NFRAME <<")" << std::endl;
    fp << "\tSteps per frame ("<< NPFRAME << ")" << std::endl;
    fp << "\tTime step (" << DT << ")" << std::endl;
    //fp << "\tNumber of particles ("<< particles.size() << ")" << std::endl;
    fp << "\tParticle size ("<< H << ")" << std::endl;
    fp << "\tReference density ("<< REST_DENS << ")" << std::endl;
    fp << "\tGas Constant ("<< GAS_CONST << ")" << std::endl;
    fp << "\tDynamic viscosity ("<< VISC << ")" << std::endl;
    fp << "\tGravitational strength ("<< G(1) << ")" << std::endl << std::endl;

    fp.close();
  }
  else {
    cout << "Error opening the output file." << endl;
    exit(-1);
  }

}

void write_frame_data(MyVector &particles, std::ofstream& fp)
{
    fp <<  "ZONE t=\"Boundary Data\", STRANDID=1, SOLUTIONTIME=" << t << std::endl;
      for (auto b=particles.begin(); b!=std::next(particles.begin(),bound_parts); ++b)
		{
			//Vector2d a=  p.f/p.rho;
	        fp << b->x(0) << " " << b->x(1) << " ";
	        fp << b->v(0) << " " << b->v(1) << " ";
	        fp << b->vh(0) << " " << b->vh(1) << " ";
	        fp << "0" << " " << "0" << std::endl;
      }

    fp <<  "ZONE t=\"Particle Data\", STRANDID=2, SOLUTIONTIME=" << t << std::endl;
      for (auto p=std::next(particles.begin(),bound_parts+1); p!=particles.end(); ++p)
		{
			Eigen::Vector2d a=  p->f/p->rho;
	        fp << p->x(0) << " " << p->x(1) << " ";
	        fp << p->v(0) << " " << p->v(1) << " ";
	        fp << p->vh(0) << " " << p->vh(1) << " ";
	        fp << a(0) << " " << a(1) << std::endl;
      }
}

int main(int argc, char *argv[]) 
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	MyVector particles;	

	InitSPH(particles);
	my_kd_tree_t mat_index(2,particles,10);
	mat_index.index->buildIndex();
	
	//query points;
	
	// std::vector<std::pair<size_t,double>> matches;
	// const size_t nMatches= mat_index.index->radiusSearch(&particles[200+bound_parts].x[0], search_radius, matches, params);	
	// cout << "radiusSearch(): radius=" << sqrt(search_radius) << " -> " << nMatches << " matches\n";
	// for (size_t i = 0; i < nMatches; i++) {
	// 	cout << "idx["<< i << "]=" << matches[i].first << " dist["<< i << "]=" << sqrt(matches[i].second) << endl;
	// 	cout << particles[matches[i].first].x[0] << "\t" << particles[matches[i].first].x[1] << endl; 
	// }
	// cout << "\n";

	char cwd[1024];
	getcwd(cwd, sizeof(cwd));
	//Add in the output folder to the path.
	strcat(cwd,"/Output/");
	write_settings(cwd);

	std::stringstream str;
	str << cwd << "Test.plt";
	std::ofstream fp(str.str().c_str(), std::ios::out);

	if(fp.is_open()) {
		
		cout << "Starting simulation..." << endl;
		//Write file header defining veriable names
		fp <<  "VARIABLES = x, y, Vx, Vy, Vx_half, Vy_half, ax, ay" << std::endl;
		


		switch (integrator) 
		{
			case 'E' :
				Step_1(particles, mat_index);
				write_frame_data(particles, fp);
				for (int frame = 1; frame<= NFRAME; ++frame) {
				  cout << "Frame Number: " << frame << endl;
				  for (int i=0; i< NPFRAME; ++i) {
				    Update_Euler(particles, mat_index);
				  }
				   write_frame_data(particles, fp);
				}
			case 'L' :
				Step_1(particles, mat_index);
				write_frame_data(particles, fp);

				for (int frame = 1; frame<= NFRAME; ++frame) {
				  cout << "Frame Number: " << frame << endl;
				  for (int i=0; i< NPFRAME; ++i) {
				    Update_leapfrog(particles, mat_index);
				  }
				  
				  write_frame_data(particles, fp);
				}
				
		}
	  
	}
	else {
		cout << "Error opening frame output file." << endl;
		exit(-1);
	}
	fp.close();

	cout << "Simulation complete. Output is available in /Output!" << endl;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2-t1).count();
    cout << "Time taken:\t" << duration/1e06 << " seconds" << endl;
	
	return 0;
}