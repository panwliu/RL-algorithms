IF (LINUX)
  add_executable(mujocosim_exe mujocosim_exe.cpp)
  set_property(TARGET mujocosim_exe PROPERTY CXX_STANDARD 17)
  target_link_libraries(mujocosim_exe
    ${LIBMUJOCO200}
    ${OPENGL_LIBRARY}
    ${LIBGLEW}
    ${LIBGLFW}
    )

ELSEIF (APPLE)
  # add_executable(mujocosim_exe mujocosim_exe.cpp)
  # set_property(TARGET mujocosim_exe PROPERTY CXX_STANDARD 17)
  # target_link_libraries(mujocosim_exe
  #   ${LIBMUJOCO200}
  #   ${OPENGL_LIBRARY}
  #   # ${LIBGLEW}
  #   ${LIBGLFW}
  #   )

  # add_executable(mujocosim mujocosim.cpp)
  # set_property(TARGET mujocosim PROPERTY CXX_STANDARD 17)
  # target_link_libraries(mujocosim
  #   ${LIBMUJOCO200}
  #   ${OPENGL_LIBRARY}
  #   # ${LIBGLEW}
  #   ${LIBGLFW}
  #   )

  pybind11_add_module(mujocosim mujocosim.cpp)
  target_link_libraries(mujocosim
    PUBLIC ${LIBMUJOCO200}
    PUBLIC ${OPENGL_LIBRARY}
    PUBLIC ${LIBGLFW}
    )
ENDIF()

