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

// ----------- Mujoco Visualizer -----------
MujocoVis::MujocoVis(MujocoSim& sim)
{
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
        .def("step", &MujocoSim::step);
        
    py::class_<MujocoVis>(m, "MujocoVis")
        .def(py::init<MujocoSim& >())
        .def("render", &MujocoVis::render)
        .def("window_should_close", &MujocoVis::window_should_close);
}