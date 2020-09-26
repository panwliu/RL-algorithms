#include "mujocosim.h"

// ----------- Mujoco Simulator -----------
MujocoSim::MujocoSim(char* _model_name)
{
    init_sim(_model_name);
}

bool MujocoSim::init_sim(char* _model_name)
{
    const char* mjkey_path = getenv("MJKEY_PATH");
    if ( mjkey_path == NULL)
    {
        std::cout<<"Set MJKEY_PATH for mjkey.txt"<<std::endl;
        return false;
    }

    mj_activate(mjkey_path);

    char error[1000];
    m_ = mj_loadXML(_model_name, NULL, error, 1000);
    if (!m_)
    {
        std::cout<<error<<std::endl;
        return false;
    }

    d_ = mj_makeData(m_);

    std::cout<<"# of qpos:\t"<< m_->nq << std::endl;
    std::cout<<"# of qvel:\t"<< m_->nv << std::endl;
    std::cout<<"# of act :\t"<< m_->nu << std::endl;
    std::cout<<"# of body:\t"<< m_->nbody << std::endl;
    std::cout<<"# of joint:\t"<< m_->njnt << std::endl;

    return true;
}

mjModel* MujocoSim::model()
{
    return m_;
}

mjData* MujocoSim::data()
{
    return d_;
}

mjtNum MujocoSim::time()
{
    return d_->time;
}

py::array_t<double> MujocoSim::state()
{
    int nq = m_->nq;
    int nv = m_->nv;

    auto state_array = py::array_t<double>(nq+nv);
    py::buffer_info buffer = state_array.request();
    double* buffer_ptr = (double* ) buffer.ptr;
    
    for (int i=0; i<nq; i++)
        buffer_ptr[i] = d_->qpos[i];
    for (int i=0; i<nv; i++)
        buffer_ptr[i+nq] = d_->qvel[i];

    return state_array;
}

py::array_t<double> MujocoSim::control()
{
    int nu = m_->nu;

    auto control_array = py::array_t<double>(nu);
    py::buffer_info buffer = control_array.request();
    double* buffer_ptr = (double* ) buffer.ptr;
    
    for (int i=0; i<nu; i++)
        buffer_ptr[i] = d_->ctrl[i];

    return control_array;
}

py::array_t<double> MujocoSim::step(py::array_t<double> action)
{
    int nu = m_->nu;
    py::buffer_info action_buffer = action.request();
    double* action_buffer_ptr = (double* ) action_buffer.ptr;

    if (action_buffer.size != nu)
    {
        std::cout<<"Action size incorrect! Should be array of size "<<nu<<" ..."<<std::endl;
        return state();
    }
    
    for (int i=0; i<nu; i++)
        d_->ctrl[i] = action_buffer_ptr[i];

    mjtNum simstart = time();
    while( time() - simstart < 1.0/60.0 )
        mj_step(m_, d_);

    return state();
}

py::array_t<double> MujocoSim::reset(py::array_t<double> qpos_d)
{
    int nq = m_->nq;

    py::buffer_info buffer = qpos_d.request();
    double* buffer_ptr = (double* ) buffer.ptr;

    if (buffer.size == 0)
    {
        mj_resetData(m_, d_);
        mj_forward(m_, d_);
    }
    else if (buffer.size != nq)
    {
        std::cout<<"State size incorrect! Should be array of size "<<nq<<" ..."<<std::endl;
    }
    else
    {
        mj_resetData(m_, d_);
        mj_forward(m_, d_);
        for (int i=0; i<nq; i++)
            d_->qpos[i] = buffer_ptr[i];
        mj_forward(m_, d_);
    }

    return state();
}

// ----------- Mujoco Visualizer -----------
// keyboard callback
void MujocoVis::keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(this_ptr->m_, this_ptr->d_);
        mj_forward(this_ptr->m_, this_ptr->d_);
    }
}

// mouse button callback
void MujocoVis::mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    this_ptr->button_left_ =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    this_ptr->button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    this_ptr->button_right_ =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &(this_ptr->lastx_), &(this_ptr->lasty_));
}

// mouse move callback
void MujocoVis::mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !(this_ptr->button_left_) && !(this_ptr->button_middle_) && !(this_ptr->button_right_) )
        return;

    // compute mouse displacement, save
    double dx = xpos - this_ptr->lastx_;
    double dy = ypos - this_ptr->lasty_;
    this_ptr->lastx_ = xpos;
    this_ptr->lasty_ = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( this_ptr->button_right_ )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( this_ptr->button_left_ )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(this_ptr->m_, action, dx/height, dy/height, &(this_ptr->scn_), &(this_ptr->cam_));
}

// scroll callback
void MujocoVis::scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(this_ptr->m_, mjMOUSE_ZOOM, 0, -0.05*yoffset, &(this_ptr->scn_), &(this_ptr->cam_));
}

MujocoVis* MujocoVis::this_ptr = NULL;

MujocoVis::MujocoVis(MujocoSim& sim)
{
    this_ptr = this;

    m_ = sim.model();
    d_ = sim.data();

    if( !glfwInit() )
        mju_error("Could not initialize GLFW");
    window_ = glfwCreateWindow(1200, 900, "MujocoSim", NULL, NULL);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    mjv_makeScene(m_, &scn_, 1000);
    mjr_makeContext(m_, &con_, mjFONTSCALE_100);

    glfwSetKeyCallback(window_, MujocoVis::keyboard);
    glfwSetCursorPosCallback(window_, MujocoVis::mouse_move);
    glfwSetMouseButtonCallback(window_, MujocoVis::mouse_button);
    glfwSetScrollCallback(window_, MujocoVis::scroll);
}

bool MujocoVis::window_should_close()
{
    return glfwWindowShouldClose(window_);
}

void MujocoVis::render()
{
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    mjv_updateScene(m_, d_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    glfwSwapBuffers(window_);

    glfwPollEvents();
}


// ----------- main -----------
int main()
{

    MujocoSim sim((char* ) "../models/cartpole.xml");
    // MujocoVis viewer(sim.model(), sim.data()); 
    MujocoVis viewer(sim); 

    while( !viewer.window_should_close() )
    {
        // sim.step();
        viewer.render();
    }

    return 0;

}

// ----------- bindings -----------
PYBIND11_MODULE(mujocosim, m) {
    py::class_<MujocoSim>(m, "MujocoSim")
        .def(py::init<char* >())
        .def("time", &MujocoSim::time)
        .def("state", &MujocoSim::state)
        .def("control", &MujocoSim::control)
        .def("step", &MujocoSim::step)
        .def("reset", &MujocoSim::reset);
        
    py::class_<MujocoVis>(m, "MujocoVis")
        .def(py::init<MujocoSim& >())
        .def("render", &MujocoVis::render)
        .def("window_should_close", &MujocoVis::window_should_close);
}