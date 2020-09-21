cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(rll)

IF (LINUX)
  set(MUJOCO_INCLUDE_DIR ~/.mujoco/mujoco200_linux/include)

  include_directories(
    ${MUJOCO_INCLUDE_DIR}
  )

  find_library(LIBMUJOCO200 libmujoco200.so ~/.mujoco/mujoco200_linux/bin)
  find_package(OpenGL)
  find_library(LIBGLEW libglew.so ~/.mujoco/mujoco200_linux/bin)
  find_library(LIBGLFW libglfw.so.3 ~/.mujoco/mujoco200_linux/bin)

ELSEIF (APPLE)
  set(MUJOCO_INCLUDE_DIR ~/.mujoco/mujoco200_macos/include)

  include_directories(
    ${MUJOCO_INCLUDE_DIR}
  )

  find_library(LIBMUJOCO200 libmujoco200.dylib ~/.mujoco/mujoco200_macos/bin)
  find_package(OpenGL)
  find_library(LIBGLEW libglew.so ~/.mujoco/mujoco200_macos/bin)
  find_library(LIBGLFW libglfw.3.dylib ~/.mujoco/mujoco200_macos/bin)

ENDIF()

add_subdirectory(extern/pybind11)
add_subdirectory(src)



