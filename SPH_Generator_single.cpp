/* Smoothed Particle Hydrodynamics (SPH) Generator */
/* Test Code for single thread (Functional)*/

#include <iostream>
#include <cmath>
#include <fstream>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <dirent.h>
#include <chrono>
#include <windows.h>
#include <sstream>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

using std::cout;
using std::cin;
using std::endl;
using std::ofstream;
using std::string;
using namespace std::chrono;

#define VERSION_TAG "SPHView00"

/*** SYSTEM PARAMETERS ***/
typedef struct sim_param_t {
  string fname;    //File name
  int simtype; //Type of simulation: 0 if box, 1 if circle
  bool sizeorcount; //true if the size is defined. false if number is defined
  int nframes;    //Number of frames
  int npframe;    //Steps per frame
  float h;        //Paricle size
  float dt;       //Time step
  float rho0;     //Reference Density
  float k;        //Bulk Modulus
  float mu;       //Viscosity
  float g;        //Gravity strength
  int n;          //Number of particles
} sim_param_t;

/*** SYSTEM STATE ***/

typedef struct sim_state_t {
  int n;               /* Number of particles */
  float mass;          /* Partile mass */
  float* rho;          /* Densities */
  /* For vectors, (x[i+0] = x-component, x[i+1]= y-component) */
  float* x;            /* Positions */
  float* vh;           /* Velocities (Half Step) */
  float* v;            /* Velocities (Full Step) */
  float* a;            /* Acceleration */
} sim_state_t;



static void default_params(sim_param_t* params) 
{
  params->fname = ("run.plt");
  params->simtype=1;
  params->sizeorcount=true;
  params->nframes = 200;
  params->npframe = 100;
  params->dt = 1e-4;
  params->h = 5e-2;
  params->rho0 = 1000;
  params->k = 1e3;
  params->mu = 0.1;
  params->g = 9.81;
  params->n=0;

}

static std::string find_num_value(string line) 
{
  std::size_t pos = line.find_first_of("0123456789.");
  std::string str=line.substr(pos);
  return(str);
}


/*PRINT THE PARAMETERS USED TO THE TERMINAL */
static void print_usage(sim_param_t* param) 
{
  //sim_param_t param;
  //default_params(&param);

  std::cout << "\t Output file name:\t" << param->fname.c_str() << std::endl;
  std::cout << "\t Number of frames:\t" << param->nframes << std::endl;
  std::cout << "\t Steps per frame:\t" << param->npframe << std::endl;
  std::cout << "\t Time step:\t" << param->dt << std::endl;
  std::cout << "\t Particle size:\t" << param->h << std::endl;
  std::cout << "\t Reference density:\t" << param->rho0 << std::endl;
  std::cout << "\t Bulk modulus:\t" << param->k << std::endl;
  std::cout << "\t Dynamic viscosity:\t" << param->mu << std::endl;
  std::cout << "\t Gravitational strength:\t" << param->g << std::endl;
  std::cout << "\t Simulation Type (0 = Box, 1 = Circle):\t" << param->simtype << std::endl;
}

int get_params(char *argv[], sim_param_t* params) 
{

 //Input file arguments to define the parameter
 std::string hash= "#";
 std::string dp = "default_params? ";
 std::string yes= "yes";
 std::string no= "no";
 std::string filename = "outfile ";
 std::string nframe = "nframes ";
 std::string npframe = "npframe ";
 std::string dt = "dt ";
 std::string npart = "nparticles ";
 std::string psize = "psize ";
 std::string rho0 = "rho_ref ";
 std::string Bmod = "Bmod ";
 std::string mu = "viscosity ";
 std::string g = "gravity ";
 std::string simtype = "simtype ";
 bool defp=true, part=false, size=false;
 //std::string  = "outfile";

//Open the file using ifstream
   std::ifstream afile (argv[1], std::ios::in);
   //Check that the file has been opened successfully
 if (afile.is_open()) {
   std::string line;

   while (std::getline(afile, line)) {
     //std::cout << line << '\n';
     //Check for comment lines
     if(line=="") {

     }
     else if(line.find(hash)!=std::string::npos) {
       //cout << "This is a comment" << endl;
     }
     else if(line.find(dp)!=std::string::npos){
       //Check if default parameters are being used. If yes, run default_params.
       cout << "this is the default parameter line" << endl;
       if(line.find(yes)!=std::string::npos){
         //If the file says yes, then default parameters are to be used
        default_params(params);
        cout << "Using default parameters..." << endl;
        return 0;
       }
        else if(line.find(no)!=std::string::npos) {
          //If not, then take the parameters from the input file
          cout <<"Default parameters are not being used..." << endl;
          defp=false; //Define that the default_params are not being used.

        }
        else {
          std::cerr << "Invalid option for 'default_params?' Please put 'yes' or 'no'" << endl;
          return -1;
        }
      }
      else if(defp==false) {
          if(line.find(filename)!=std::string::npos){

            std::size_t pos= line.find(filename) + 8;
            //cout << pos << endl;
            std::string str=line.substr(pos);
            //char* cstr= str.c_str();
            cout << str << endl;
            //params->fname= cstr;
            params->fname = str;
            cout << "New output filename:\t" << params->fname.c_str() << endl;
          }
          else if(line.find(nframe)!=std::string::npos){
            string str= find_num_value(line);
            params->nframes = atof(str.c_str());
            cout << "New frame number:\t" << params->nframes << endl;
          }
          else if(line.find(npframe)!=std::string::npos){
            string str= find_num_value(line);
            params->npframe = atof(str.c_str());
            cout << "New steps per frame:\t" << params->npframe << endl;
          }
          else if(line.find(dt)!=std::string::npos){
            string str= find_num_value(line);
            params->dt = atof(str.c_str());
            cout << "New timestep:\t" << params->dt << endl;
          }
          else if(line.find(rho0)!=std::string::npos){
            string str= find_num_value(line);
            params->rho0 = atof(str.c_str());
            cout << "New reference density:\t" << params->rho0 << endl;
          }
          else if(line.find(Bmod)!=std::string::npos){
            string str= find_num_value(line);
            params->k = atof(str.c_str());
            cout << "New fbulk modulus:\t" << params->k << endl;
          }
          else if(line.find(mu)!=std::string::npos){
            string str= find_num_value(line);
            params->mu = atof(str.c_str());
            cout << "New dynamic viscosity:\t" << params->mu << endl;
          }
          else if(line.find(g)!=std::string::npos){
            string str= find_num_value(line);
            params->g = atof(str.c_str());
            cout << "New gravity strength:\t" << params->g << endl;
          }
          else if(line.find(simtype)!=std::string::npos){

            if(line.find("Box")!=std::string::npos){
              params->simtype=0;
            }
            else if(line.find("Circle")!=std::string::npos) {
              params->simtype=1;
            }
            else {
              cout << endl << "****************************************************************************" << endl ;
              cout << endl << "Simulation type not defined correctly. Assuming default value of 'Box' simulation." <<endl;
              cout << endl << "****************************************************************************" << endl;
              params->simtype=0;
            }

            cout << "New simulation type:\t" << params->simtype << endl;
          }
          //In need of work....
          else if(line.find(npart)!=std::string::npos){
            if (size == true) {
              cout << endl << "****************************************************************************" << endl ;
              cout << endl << "Over-constrained starting conditions." <<endl;
              cout << "Please check that only particle #, or particle diameter, is defined." << endl;
              cout << endl << "****************************************************************************" << endl;
              exit(-1);
            }
            else {
              string str= find_num_value(line);
              params->sizeorcount=false;
              params->n = atoi(str.c_str());
              cout << "New number of particles:\t" << params->n << endl;
              //Some function to find hh...
              part = true;
            }
          }
          else if(line.find(psize)!=std::string::npos){
            if (part== true) {
              cout << endl << "*****************************************************************************" << endl;
              cout << endl << "Over-constrained starting conditions." <<endl;
              cout << "Please check that only particle #, or particle diameter, is defined." << endl;
              cout << endl << "*****************************************************************************" << endl;
              exit(-1);
            }
            else {
              string str= find_num_value(line);
              params->sizeorcount=true;
              params->h = atof(str.c_str());
              size = true;
              cout << "New particle diameter:\t" << params->h << endl;
            }
          }
        }
        else {
          cout << "It has not been specified whether default parameters are to be used or not." << endl;
          cout << "Please check that 'default_params?' has been defined in the input file using 'yes' or 'no'." << endl;
          return -1;
        }
   }
   afile.close();
   print_usage(params);
 }
 else {
   std::cerr << "Unable to open file\n";
   exit(-1);
 }

  return 0;
} //End of get_params


/*** DENSITY CALCULATION ***/
void compute_density(sim_state_t* s, sim_param_t* params) 
{
  //Bring over the variables from the starting parameters
  int n = s->n, i, j;
  float* rho = s->rho;
  const float* x=s->x;
  float h = params->h;
  float h2= h*h;
  float h8 = (h2*h2)*(h2*h2);
  float C= 4* s->mass /M_PI/h8;



  for (i=0; i< n; ++i) {
    rho[i] += 4* s->mass /M_PI/h2;
    for (j=i+1; j<n; ++j) {
      float dx= x[2*i+0]-x[2*j+0]; //X-direction
      float dy= x[2*i+1]-x[2*j+1]; //Y-direction
      float r2=dx*dx +dy*dy;
      float z=h2-r2;

      if (z>0) {
        float rho_ij = C*z*z*z;
        rho[i]+=rho_ij;
        rho[j]+=rho_ij;
      }

    }
  }
}

/*** ACCELERATION CALCULATION ***/
void compute_acceleration(sim_state_t* s, sim_param_t* params)
{

  // Unpack parameters
  const float h= params->h;
  const float rho0 = params->rho0;
  const float k = params->k;
  const float mu = params->mu;
  const float g = params->g;
  const float mass = s->mass;
  const float h2= h*h;

  //unpack the system state
  const float* rho= s->rho;
  const float* x= s->x;
  const float* v = s->v;
  float* a= s ->a;
  int n =s ->n;

  //Compute the density
  compute_density(s,params);

  // Begin with gravty and surface forces (for now no surface force)
  int i, j;
  for (i=0; i<n; ++i) {
    a[2*i+0]=0; //x-direction
    a[2*i+1]=-g; //y-direction
  }

  //constants for interaction term
  float C0=mass/M_PI/(h2*h2), Cp=15*k, Cv= -40*mu;
  // Compute the particle interations
  for (i=0 ; i<n ;++i) {
    const float rhoi= rho[i];
    for (j=i+1; j<n; ++j) {
      float dx= x[2*i+0]-x[2*j+0]; //X-direction
      float dy= x[2*i+1]-x[2*j+1]; //Y-direction
      float r2=dx*dx +dy*dy;
      //Check if the particles are inside each other (i.e. pushing each other)
      if (r2<h2) {
        const float rhoj = rho[j];
        float q= sqrt(r2)/h;
        float u= 1-q;
        float w0 = C0*u/rhoi/rhoj;
        float wp = w0*Cp* (rhoj+rhoj-2*rho0)*u/q;
        float wv = w0*Cv;
        float dvx= v[2*i+0]-v[2*j+0]; //X-velocity
        float dvy= v[2*i+1]-v[2*j+1]; //Y-velocity
        a[2*i+0] += (wp*dx + wv*dvx);
        a[2*i+1] += (wp*dy + wv*dvy);
        a[2*j+0] -= (wp*dx + wv*dvx);
        a[2*j+1] -= (wp*dy + wv*dvy);
      }
    }
  }
}

/*** BOUNDARY CONDITIONS ***/

static void damp_reflect(int which, float barrier, float* x, float* v, float* vh)
{
  //Coefficient of restitution
  const float DAMP=0.75;

  // Ignore wallriders
  if (v[which]==0)
    return;

  //Find the distance travelled past barrier based on time from collision
  float tbounce = (x[which]-barrier)/v[which];
  x[0]-=v[0]*(1-DAMP)*tbounce;
  x[1]-=v[1]*(1-DAMP)*tbounce;

  //Reflect position and velocity
  x[which] = 2*barrier-x[which];
  v[which] = -v[which];
  vh[which] = -vh[which];

  //Damp the velocities
  v[0] *=DAMP; v[1] *=DAMP;
  vh[0]*=DAMP; vh[1]*=DAMP;

}

/* Reflection boundary conditions*/
static void reflect_bc(sim_state_t* s) 
{
  //Domain Boundary
  const float Xmin = 0.0, Xmax = 1.0, Ymin = 0.0, Ymax = 1.0;

  float* vh=s->vh;
  float* v=s->v;
  float* x=s->x;
  int n=s->n, i;
  for (i=0; i<n; ++i, x+=2, v+=2, vh+=2) {
    if (x[0]<Xmin) damp_reflect(0,Xmin, x,v,vh);
    if (x[0]>Xmax) damp_reflect(0,Xmax, x,v,vh);
    if (x[1]<Ymin) damp_reflect(1,Ymin, x,v,vh);
    if (x[1]>Ymax) damp_reflect(1,Ymax, x,v,vh);

  }
}

/*** LEAPFROG TIME-STEP INTEGRATION ***/
// The velocity is updated at half-steps, and positions at full-steps.
// To get v^(i+1), compute another half-step from v^(i+1/2), using a^i
// At t=0, there's no vh, so perform a half step to get vh[0].
void leapfrog_start(sim_state_t* s, double dt) 
{
  const float* a = s->a;
  float* vh= s->vh;
  float* v = s->v;
  float* x = s->x;
  int n=s->n, i;
  for (i=0; i <2*n; ++i) vh[i] = v[i] + a[i]*dt/2;
  for (i=0; i <2*n; ++i) v[i] += a[i]*dt;
  for (i=0; i <2*n; ++i) x[i] += vh[i]*dt;
  reflect_bc(s);
}

void leapfrog_step(sim_state_t* s, double dt) 
{
  const float* a = s->a;
  float* vh= s->vh;
  float* v = s->v;
  float* x = s->x;
  int n=s->n, i;
  for (i=0; i <2*n; ++i) vh[i]+= a[i]*dt;
  for (i=0; i <2*n; ++i) v[i]  = vh[i]+ a[i]*dt/2;
  for (i=0; i <2*n; ++i) x[i] += vh[i]*dt;
  reflect_bc(s);
}

/*** DEFINE THE STARTING GEOMETRY OF THE POPULATED AREA ***/
typedef int (*domain_fun_t)(float,float) ; //Define the function type

//If a box is wanted, return if less than 0.5 square
int box_indicator(float x, float y) 
{
  return (x<0.5) && (y<0.5) ;
}

//For an ellipse, return if inside the radius indicated.
int circ_indicator(float x, float y) 
{
  float dx= (x-0.5);
  float dy= (y-0.3);
  float r2=dx*dx+dy*dy;
  //std::cout<<"R^2:" << r2 << std::endl;
  return (r2<0.25*0.25);
}

int find_count(sim_param_t* param, domain_fun_t indicatef) 
{
  float h= param->h;
  float hh=h/1.3; //Mesh size to place particles upon

  //Count mesh points that fall in the indicated region
  int count = 0;
  for (float x=0; x<1; x+=hh)
    for (float y=0; y<1; y+=hh) {
  //  std::cout << "Indicator Fuction Return: "<< indicatef(x,y) << std::endl;
    count += indicatef(x,y);
  }
    //if the mesh point is inside the starting area, indicatef returns 1. 0 if not.
    return count;
}

double find_hh(int n, sim_param_t* param, domain_fun_t indicatef) 
{
  // Do some stuff to find the spacing
  float hh;
  hh= 5e-2/1.3;
  return hh;
}

/*** PLACE PARTICLES INSIDE POPULATED AREA***/
//indicatef is defined (box_indicator or circ_indicator) to which start is wanted
sim_state_t* place_particles(sim_param_t* param, domain_fun_t indicatef) 
{
  //Initilize the state structure and allocate memory
  sim_state_t* s =new sim_state_t();
  int count= find_count(param,indicatef);
  float hh=param->h/1.3;


  if(param->sizeorcount==false) {
      count=param->n;
      hh= find_hh(count,param,indicatef);
  }
  //cout << count << "   " << hh << endl;
  s->n=count;
  s->x = (float*)calloc(2*count,sizeof(float));
  s->v= (float*)calloc(2*count, sizeof(float));
  s->rho= (float*)calloc(2*count, sizeof(float));
  s->a= (float*)calloc(2*count, sizeof(float));
  s->vh= (float*)calloc(2*count, sizeof(float));
  int p=0;
  for (float x=0; x<1; x+=hh) {
    for (float y=0; y<1; y+=hh) {

      if (indicatef(x,y)) {
        //std::cout << "x: "<< x << " y: " << y << " " << std::endl;
        s->x[2*p+0]=x;
        s->x[2*p+1]=y;
        ++p;
        //std::cout<< "p: " << p << std::endl;
      }
    }
  }

  return s;
}

/*** NORMALISE THE MASS TO THE REFERENCE DENSITY ***/
void normalise_mass(sim_state_t* s, sim_param_t* param) 
{
  s->mass=1; //Define the mass as 1, then find the right particle density
  compute_density(s,param);
  //s->n=;
  float rho0= param->rho0;
  float rho2s=0;
  float rhos=0;
  for (int i=0; i<s->n; ++i) {
    rho2s +=(s->rho[i])*(s->rho[i]);
    rhos +=s->rho[i];
  }
  s->mass *=(rho0*rhos/rho2s);
}


sim_state_t* init_particles(sim_param_t* param) 
{
  sim_state_t* s;
//Check the simulation type and do the right action.
  if(param->simtype==0) s=place_particles(param, box_indicator);
  else if(param->simtype==1) s=place_particles(param, circ_indicator);
  else {
     cout << "Simtype not defined correctly." << endl;
     exit(-1);
    }
  normalise_mass(s,param);
  return s;
}

// Check that the particles have not left the boundary (Simulation has not gone beserk)
void check_state(sim_state_t* s) 
{
  for (int i=0; i<s->n; ++i) {
    float xi= s->x[2*i+0];
    float yi= s->x[2*i+1];
    assert( xi >= 0 || xi <= 1);
    assert( yi >= 0 || yi <= 1);
  }
}


static void write_settings(int n, sim_param_t* param) 
{
  //Find current working directory
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  strcat(cwd,"/Output/");
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
  str.append(param->fname);
  str.append("_settings.txt"); 
  //insert a settings file to show the simulation settings.
  //cout << str << endl;
  std::ofstream fp(str.c_str(), std::ios::out);
  //std::ofstream fp("Test_Settings.txt", std::ios::out);
  if(fp.is_open()) {
    //const float mass = s->mass;
    fp << "VERSION: " << VERSION_TAG << std::endl << std::endl; //Write version
    fp << "SIMULATION PARAMETERS:" << std::endl; //State the system parameters.
    fp << "\tNumber of frames (" << param->nframes <<")" << std::endl;
    fp << "\tSteps per frame ("<< param->npframe << ")" << std::endl;
    fp << "\tTime step (" << param->dt << ")" << std::endl;
    fp << "\tNumber of particles ("<< n << ")" << std::endl;
    fp << "\tParticle size ("<< param->h << ")" << std::endl;
    fp << "\tReference density ("<< param->rho0 << ")" << std::endl;
    fp << "\tBulk modulus ("<< param->k << ")" << std::endl;
    fp << "\tDynamic viscosity ("<< param->mu << ")" << std::endl;
    fp << "\tGravitational strength ("<< param->g << ")" << std::endl << std::endl;

    fp.close();
  }
  else {
    cout << "Error opening the output file." << endl;
    exit(-1);
  }

}


void write_frame_data(string fname, int n,sim_state_t* state,int frame) 
{
  //Open the file in an append mode, so each frame is written to 1 file
  //Open the file in an overwrite mode, to start the file header.
 
  //Find current working directory
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  //Add in the output folder to the path.
  strcat(cwd,"/Output/");
  std::stringstream str;
  str << cwd << "Test" << frame << ".plt";
  std::ofstream fp(str.str().c_str(), std::ios::out);
  if(fp.is_open()) {
   // fp <<  "\tFrame: " << frame << std::endl;
    fp <<  "VARIABLES = x, y, V_x, V_y, V_x_half, V_y_half, a_x, a_y" << std::endl;

      for(int i=0; i<n; ++i) {
        fp << state->x[2*i+0] << " " << state->x[2*i+1] << " ";
        fp << state->v[2*i+0] << " " << state->v[2*i+1] << " ";
        fp << state->vh[2*i+0] << " " << state->vh[2*i+1] << " ";
        fp << state->a[2*i+0] << " " << state->a[2*i+1] << std::endl;
      }
      fp.close();
  }
  else {
    cout << "Error opening frame output file." << endl;
    exit(-1);
  }
}


int main(int argc, char *argv[]) 
{
  //std::cout << argv[1] << std::endl;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  sim_param_t params;
  int frame=0;
  //Check that parameters are retrieved successfully
  if (get_params(argv, &params) !=0)
   exit(-1);
  // Initialise the state parameters
  sim_state_t* state = init_particles(&params);

  int nframes = params.nframes;
  int npframe = params.npframe;
  float dt = params.dt;
  int n = state->n;


  write_settings(n, &params);
  write_frame_data(params.fname,n,state, frame);
  cout << "\tParticle Number:\t" << n << endl;
  cout << "Starting Simulation..." << endl;
  compute_acceleration(state, &params);
  leapfrog_start(state, dt);
  check_state(state);

  for (int frame = 1; frame<= nframes; ++frame) {
    cout<< "Frame Number: " << frame << std::endl;
    for (int i=0; i< npframe; ++i) {
      compute_acceleration(state, &params);
      leapfrog_step(state, dt);
      check_state(state);
    }
    write_frame_data(params.fname, n, state,frame);
  }

  cout << "Simulation complete. Output is available in /Output!" << endl;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2-t1).count();
  cout << "Time taken:\t" << duration/1e06 << " seconds" << endl;
  return(0);
}
