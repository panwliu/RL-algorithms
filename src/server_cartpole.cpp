# include "server_cartpole.h"

namespace rll{

ServerCartpole::ServerCartpole(int _port_self, int _port_remote)
{
    std::cout<<"====== Enter Server Cartpole ======"<<std::endl;
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    memset(&sockaddr_local_, 0, sizeof(sockaddr_local_));
    sockaddr_local_.sin_family = AF_INET;
    sockaddr_local_.sin_addr.s_addr = htonl(INADDR_ANY);//inet_addr("127.0.0.1");
    sockaddr_local_.sin_port = htons(_port_self);
    
    if (bind(fd_, (struct sockaddr*)&sockaddr_local_, sizeof(sockaddr_local_))<0)
        std::cout<<"Socket binding failed"<<std::endl;
    else
        std::cout<<"Socket binding succeed"<<std::endl;

    memset(&sockaddr_remote_, 0, sizeof(sockaddr_remote_));
    sockaddr_remote_.sin_family = AF_INET;
    sockaddr_remote_.sin_addr.s_addr = htonl(INADDR_ANY);
    sockaddr_remote_.sin_port = htons(_port_remote);
    sockaddr_remote_len_ = sizeof(sockaddr_remote_);


    server_config_.SetSdfFile("/home/pan/rll/worlds/cartpole_world.sdf");
    server_config_.SetUpdateRate(1000);

    // server_ = std::make_unique<ignition::gazebo::Server>(server_config_);
    // server_->SetUpdatePeriod(std::chrono::milliseconds(1));       // 1ms is default
    // std::cout<<"server update rate: "<<*server_config_.UpdateRate()<<std::endl;
    // std::cout<<" server update period: "<<(*server_config_.UpdatePeriod()).count()<<std::endl;
    // server_->Run();

    states_topic_ = "/model/cartpole/states";
    commands_topic_ = "/model/cartpole/commands";
    node_.Subscribe(states_topic_, &ServerCartpole::StateTopicCB, this);
    commands_pub_ = node_.Advertise<ignition::msgs::Float_V>(commands_topic_);
}

void ServerCartpole::recvFromRL()
{
    // recvfrom is blocking
    int len = recvfrom(fd_, commands_, sizeof(commands_), 0, (struct sockaddr*)&sockaddr_remote_, &sockaddr_remote_len_);

    if (len<0)
        return;

    if (commands_[1] == 21)
        server_->Run(/*blocking=*/true, /*iterations=*/1, /*paused=*/false);
    else if (commands_[1] == 22)
        server_ = std::make_unique<ignition::gazebo::Server>(server_config_);
    else
        sendCommands();
    
}

void ServerCartpole::sendCommands()
{
    commands_msg_.clear_data();
    for (int k=0; k<msg_len_; k++)
        commands_msg_.add_data(commands_[k]);

    commands_pub_.Publish(commands_msg_);
    
}

void ServerCartpole::StateTopicCB(const ignition::msgs::Float_V &_msg)
{
    //pauseSim();

    auto data = _msg.data();
    int id = data.Get(0);
    int type = data.Get(1);
    float t = data.Get(2);

    /*std::cout<<"id: "<<id<<" type: "<<type<<"time: "<<t<<std::endl;
    std::cout<<"Receive data from state topic ";
    for(int k=3; k<msg_len_; k++)
        std::cout<<data.Get(k)<<", ";
    std::cout<<std::endl;*/

    float state[msg_len_];
    for(int k=0; k<msg_len_; k++)
        state[k] = data.Get(k);

    sendto(fd_, state, sizeof(state),0, (struct sockaddr*)&sockaddr_remote_, sockaddr_remote_len_);
    
}

void ServerCartpole::pauseSim()
{
    server_->SetPaused(true);
}

void ServerCartpole::unpauseSim()
{
    server_->SetPaused(false);
}

}

int main()
{
    rll::ServerCartpole server(18080, 18060);

    while(1)
    {
        server.recvFromRL();
    }

    return 0;
}