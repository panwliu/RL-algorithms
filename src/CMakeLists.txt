add_executable(mujocosim mujocosim.cpp)
set_property(TARGET mujocosim PROPERTY CXX_STANDARD 17)
target_link_libraries(mujocosim
  ${LIBMUJOCO200}
  ${OPENGL_LIBRARY}
  ${LIBGLEW}
  ${LIBGLFW}
  )
