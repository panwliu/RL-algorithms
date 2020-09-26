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
        py::array_t<double> reset(py::array_t<double> qpos_d);

    private:
        mjModel* m_;
        mjData* d_;
};

class MujocoVis
{
    public:
        MujocoVis(MujocoSim& sim);                  // give sim directly will result sim's deconstructor called automatically
        // MujocoVis(mjModel* _m, mjData* _d);      // python don't work well with pointers
        ~MujocoVis()
        {
            // glfwTerminate();
            mjv_freeScene(&scn_);
            mjr_freeContext(&con_);
        }
        bool window_should_close();
        void render();
        static void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
        static void mouse_button(GLFWwindow* window, int button, int act, int mods);
        static void mouse_move(GLFWwindow* window, double xpos, double ypos);
        static void scroll(GLFWwindow* window, double xoffset, double yoffset);

    private:
        mjModel* m_;
        mjData* d_;
        mjvCamera cam_;                      // abstract camera
        mjvOption opt_;                      // visualization options
        mjvScene scn_;                       // abstract scene
        mjrContext con_;
        GLFWwindow* window_;

        bool button_left_ = false, button_middle_ = false, button_right_ =  false;
        double lastx_=0, lasty_=0;

        static MujocoVis *this_ptr;
};


#endif