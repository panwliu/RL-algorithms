#ifndef MUJOCOSIM_H_
#define MUJOCOSIM_H_

#include <iostream>
#include "mujoco.h"
#include "glfw3.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class MujocoSim
{
    public:
        MujocoSim(char* _model_name);
        ~MujocoSim()
        {
            mj_deleteData(d_);
            mj_deleteModel(m_);
            mj_deactivate();
        }
        bool init_sim(char* _model_name);
        mjModel* model();
        mjData* data();
        mjtNum time();
        py::array_t<double> state();
        py::array_t<double> control();
        py::array_t<double> step(py::array_t<double> action);

    private:
        mjModel* m_;
        mjData* d_;
};

class MujocoVis
{
    public:
        MujocoVis(MujocoSim& sim);                  // give sim directly will result sim's deconstructor called automatically
        // MujocoVis(mjModel* _m, mjData* _d);      // python don't work well with pointers, alghough works in c++
        ~MujocoVis()
        {
            // glfwTerminate();
            mjv_freeScene(&scn_);
            mjr_freeContext(&con_);
        }
        bool window_should_close();
        void render();

    private:
        mjModel* m_;
        mjData* d_;
        mjvCamera cam_;                      // abstract camera
        mjvOption opt_;                      // visualization options
        mjvScene scn_;                       // abstract scene
        mjrContext con_;
        GLFWwindow* window_;
};


#endif