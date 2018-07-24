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


#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;
#define VERSION_TAG "SPHView01"

//Solver Parameters
const static Vector2d G(0.f, 10*-9.81f); /*External forces*/
const static float REST_DENS = 1000.f; /*Resting density*/
const static float GAS_CONST = 2000.f; /*Gas constant for Eq Of State*/
const static float GAMMA = 7.f; /*Gas constant for Eq Of State*/
const static float C_s = 88.5f; /*Gas constant for Eq Of State*/
const static float B = REST_DENS*pow(C_s,2.f)/GAMMA ; /*Gas constant for Eq Of State*/
const static float H= 16.f; /* Kernel Radius */
const static float HSQ = H*H; /*Kernel squared*/
const static float MASS = 65.f; /*Mass (assuming constant between particles)*/
const static float VISC = 250.f; /*Viscosity Constant*/
const static float DT = 0.0008f; /*Timestep*/
const static int NFRAME = 400;
const static int NPFRAME = 30;

//Smoothing kernels defined in Muller et al.
const static float POLY6 = 315.f/(65.f*M_PI*pow(H,9.f));
const static float SPIKY_GRAD = -45.f/(M_PI*pow(H,6.f));
const static float VISC_LAP = 45.f/(M_PI*pow(H,6.f));

//simulation parameters
const static float EPS = H; /*Boundary epsilon*/
const static float BOUND_DAMP = -0.5f;

//Particle Data structure
typedef struct Particle {
	Particle(float _x, float _y) : x(_x,_y), v(0.f,0.f), vh(0.f,0.f), f(0.f,0.f), rho(0), p(0.f) {}
	Vector2d x, v, vh, f;
	float rho, p;
}Particle;

static vector<Particle> particles;

//Interaction
const static int MAX_PARTICLES = 2500;
const static int DAM_PARTICLES = 500;
const static int BLOCK_PARTICLES = 250;


//Simulation window parameters
const static int WINDOW_WIDTH=800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5f*WINDOW_WIDTH;
const static double VIEW_HEIGHT = 1.5f*WINDOW_HEIGHT;

/*** EULER FORWARD INTEGRATION***/
void Integrate (void) 
{
	for(auto &p : particles)
	{
		// forward Euler integration
		p.v += DT*p.f/p.rho;
		p.x += DT*p.v;

		// enforce boundary conditions
		if(p.x(0)-EPS < 0.0f)
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = EPS;
		}
		if(p.x(0)+EPS > VIEW_WIDTH) 
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = VIEW_WIDTH-EPS;
		}
		if(p.x(1)-EPS < 0.0f)
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = EPS;
		}
		if(p.x(1)+EPS > VIEW_HEIGHT)
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = VIEW_HEIGHT-EPS;
		}
	}
}

/*** LEAPFROG TIME-STEP INTEGRATION ***/
// The velocity is updated at half-steps, and positions at full-steps.
// To get v^(i+1), compute another half-step from v^(i+1/2), using a^i
// At t=0, there's no vh, so perform a half step to get vh[0].
void leapfrog_start(void)
{
	for (auto &p:particles) {
		p.vh = p.v + (p.f/p.rho)*(DT/2.f);
		p.v += DT*p.f/p.rho;
		p.x += p.vh*DT;

		/*Enforce Boundary Conditions*/
		if (p.x(0)-EPS < 0.0f) 
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = EPS;
		}
		if (p.x(0)+EPS > VIEW_WIDTH) 
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = VIEW_WIDTH - EPS;
		}

		if (p.x(1)-EPS < 0.0f) 
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = EPS;
		}
		if (p.x(1)+EPS > VIEW_HEIGHT) 
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = VIEW_HEIGHT - EPS;
		}
	}
}

void leapfrog_step(void)
{
  for (auto &p:particles) {
		p.vh += (p.f/p.rho)*(DT);
		p.v   = p.vh + (DT/2.f)*p.f/p.rho;
		p.x  += p.vh*DT;

		/*Enforce Boundary Conditions */
		if (p.x(0)-EPS < 0.0f) 
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = EPS;
		}
		if (p.x(0)+EPS > VIEW_WIDTH) 
		{
			p.v(0) *= BOUND_DAMP;
			p.x(0) = VIEW_WIDTH - EPS;
		}

		if (p.x(1)-EPS < 0.0f) 
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = EPS;
		}
		if (p.x(1)+EPS > VIEW_HEIGHT) 
		{
			p.v(1) *= BOUND_DAMP;
			p.x(1) = VIEW_HEIGHT - EPS;
		}
	}

}

void ComputeDensityPressure (void) 
{ 
	for (auto &pi: particles) 
	{
		pi.rho=0.f;
		for (auto &pj: particles) 
		{
			Vector2d rij = pj.x-pi.x;
			float r2 = rij.squaredNorm();

			if(r2<HSQ) 
			{
				pi.rho +=MASS*POLY6*pow(HSQ-r2,3.f);
			}
		}
		/*Ideal Gas Law*/
		//pi.p = GAS_CONST*(pi.rho-REST_DENS);

		/*Tait's Equation*/
		pi.p = B*(pow((pi.rho/REST_DENS),GAMMA)-1);
	}
}

void ComputeForces(void) 
{
	for (auto &pi: particles) 
	{
		Vector2d fpress(0.f,0.f);
		Vector2d fvisc(0.f,0.f);
		for(auto &pj:particles) 
		{
			if(&pi==&pj)
				continue;

			Vector2d rij = pj.x-pi.x;
			float r =rij.norm();

			if (r<H) 
			{
				/*Compute pressure contribution*/
				fpress += -rij.normalized()*MASS*(pi.p+pj.p)/(2.f*pj.rho) * SPIKY_GRAD*pow(H-r,2.f);
				/*Compute Viscosity contribution*/
				fvisc += VISC*MASS*(pj.v-pi.v)/pj.rho *VISC_LAP*(H-r);
			}
		}
		Vector2d fgrav = G*pi.rho;
		pi.f = fpress + fvisc + fgrav;
	}
}

void InitSPH(void) 
{
	cout << "Initialising simulation with " << DAM_PARTICLES << " particles" << endl;
	for( float y = EPS; y < VIEW_HEIGHT - EPS*2.f; y+=EPS) 
	{
		for(float x = VIEW_WIDTH/4; x<= VIEW_WIDTH/2; x+=H)
		{
			if (particles.size() < DAM_PARTICLES) 
			{
				float jitter = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				particles.push_back(Particle(x+jitter,y));
			}
		}
	}

	ComputeDensityPressure();
	ComputeForces();
	leapfrog_start();
}

void Update(void) 
{
	ComputeDensityPressure();
	ComputeForces();
	leapfrog_step();
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
    fp << "\tNumber of particles ("<< particles.size() << ")" << std::endl;
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

void write_frame_data(float t, std::ofstream& fp)
{
    fp <<  "ZONE, STRANDID=1, SOLUTIONTIME=" << t << std::endl;
      for(auto &p: particles) {
      	Vector2d a=  p.f/p.rho*DT;
        fp << p.x(0) << " " << p.x(1) << " ";
        fp << p.v(0) << " " << p.v(1) << " ";
        fp << p.vh(0) << " " << p.vh(1) << " ";
        fp << a(0) << " " << a(1) << std::endl;
      }
}

int main(int argc, char *argv[]) 
{
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	

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
		InitSPH();
		//Write file header defining veriable names
	    fp <<  "VARIABLES = x, y, Vx, Vy, Vx_half, Vy_half, ax, ay" << std::endl;
	    float t = 0.f;
	    write_frame_data(t, fp);

	    

	    for (int frame = 1; frame<= NFRAME; ++frame) {
		  cout<< "Frame Number: " << frame << std::endl;
		  for (int i=0; i< NPFRAME; ++i) {
		    Update();
		  }
		  t +=DT*NPFRAME;
		  write_frame_data(t,fp);
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