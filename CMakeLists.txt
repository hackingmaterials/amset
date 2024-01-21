cmake_minimum_required(VERSION 3.15)
project(amset C)

option(AMSET "Option to build amset module" ON)
option(USE_OpenMP "Option to search OpenMP library" ON)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
set(CMAKE_C_FLAGS_DEBUG "-g -DLAGWARNING")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

include(GNUInstallDirs)

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/amset/__init__.py version_file)
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)"
             amset_version ${version_file})
set(MAJOR_VERSION ${CMAKE_MATCH_1})
set(MINOR_VERSION ${CMAKE_MATCH_2})
set(MICRO_VERSION ${CMAKE_MATCH_3})
set(SERIAL "${MAJOR_VERSION}.${MINOR_VERSION}.${MICRO_VERSION}")
set(SOSERIAL "1")

if(USE_CONDA_PATH)
  message(STATUS "$ENV{CONDA_PREFIX}")
  set(CMAKE_MODULE_PATH $ENV{CONDA_PREFIX})
  set(MY_INCLUDES $ENV{CONDA_PREFIX}/include ${PROJECT_SOURCE_DIR}/c)
  link_directories($ENV{CONDA_PREFIX}/lib)
else()
  set(MY_INCLUDES ${PROJECT_SOURCE_DIR}/c)
endif()

if(USE_OpenMP)
  message(STATUS "Find OpenMP library")
  find_package(OpenMP)

  if(OpenMP_FOUND)
    message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
  endif()
endif()

if(AMSET)
  find_package(BLAS REQUIRED) # set BLAS_LIBRARIES

  if(BLAS_FOUND)
    message(STATUS "BLAS libs: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS flags: ${BLAS_LINKER_FLAGS}")
  endif()

  find_package(LAPACK REQUIRED) # set LAPACK_LIBRARIES

  if(LAPACK_FOUND)
    message(STATUS "LAPACK libs: ${LAPACK_LIBRARIES}")
    message(STATUS "LAPACK flags: ${LAPACK_LINKER_FLAGS}")
  endif()
endif()

if(AMSET)
  if(BLAS_LIBRARIES MATCHES "libmkl")
    message(STATUS "MKL detected: Set C-macros MKL_LAPACKE MULTITHREADED_BLAS")
  endif()

  if(BLAS_LIBRARIES MATCHES "libopenblas")
    message(STATUS "OpenBLAS detected: Set C-macro MULTITHREADED_BLAS")
  endif()
endif()

if(AMSET)
  # Source code
  set(SOURCES_AMSET
      ${PROJECT_SOURCE_DIR}/c/amset.c
      )

  # Static link library
  add_library(amset_static STATIC ${SOURCES_AMSET})
  target_link_libraries(amset_static m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                        ${OpenMP_C_LIBRARIES})
  target_include_directories(amset_static PRIVATE ${MY_INCLUDES})
  target_compile_options(amset_static PRIVATE ${OpenMP_C_FLAGS})

  if(BLAS_LIBRARIES MATCHES "libmkl")
    target_compile_definitions(amset_static PRIVATE MKL_LAPACKE)
  endif()

  if(BLAS_LIBRARIES MATCHES "libopenblas")
    target_compile_definitions(amset_static PRIVATE)
  endif()

  set_property(TARGET amset_static PROPERTY VERSION ${SERIAL})
  set_property(TARGET amset_static PROPERTY SOVERSION ${SOSERIAL})
  set_property(TARGET amset_static PROPERTY OUTPUT_NAME amst)
  install(TARGETS amset_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/amset.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()