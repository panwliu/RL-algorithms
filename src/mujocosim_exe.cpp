#include <iostream>
#include "mujoco.h"
#include "glfw3.h"

int main()
{

const char* mjkey_path = getenv("MJKEY_PATH");

if ( mjkey_path == NULL)
{
    std::cout<<"Set MJKEY_PATH for mjkey.txt"<<std::endl;
    return 0;
}

mj_activate(mjkey_path);

char error[1000];
mjModel* m;
mjData* d;

mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;  

m = mj_loadXML("../models/cartpole.xml", NULL, error, 1000);
if (!m)
{
    std::cout<<error<<std::endl;
    return 1;
}

d = mj_makeData(m);

std::cout<<"# of qpos:\t"<< m->nq << std::endl;
std::cout<<"# of qvel:\t"<< m->nv << std::endl;
std::cout<<"# of act :\t"<< m->nu << std::endl;
std::cout<<"# of body:\t"<< m->nbody << std::endl;
std::cout<<"# of joint:\t"<< m->njnt << std::endl;

if( !glfwInit() )
    mju_error("Could not initialize GLFW");
GLFWwindow* window = glfwCreateWindow(1200, 900, "MujocoSim", NULL, NULL);
glfwMakeContextCurrent(window);
glfwSwapInterval(1);

mjv_defaultCamera(&cam);
mjv_defaultOption(&opt);
mjv_defaultScene(&scn);
mjr_defaultContext(&con);

mjv_makeScene(m, &scn, 1000);
mjr_makeContext(m, &con, mjFONTSCALE_100);

while( !glfwWindowShouldClose(window) )
{
    mjtNum simstart = d->time;
    d->ctrl[0] = 0.05;
    while( d->time - simstart < 1.0/60.0 )
        mj_step(m, d);

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);

    glfwPollEvents();
}

// glfwTerminate();
mjv_freeScene(&scn);
mjr_freeContext(&con);

mj_deleteData(d);
mj_deleteModel(m);
mj_deactivate();

return 0;

}