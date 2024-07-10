# Mesh Deformation

Project implementing a modified version of Bilaplacian mesh deformation with Eigen and libigl

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies and create a `example` binary.

## Run

From within the `build` directory just issue:

    ./example

A glfw app should launch displaying a 3D cube.

## Using other modules of libigl

This example project uses the `igl::opengl::glfw::Viewer`, therefore it requires
the glfw module of libigl. This shows up in the CMakeLists.txt 

```cmake
igl_include(glfw)
…
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
```

Suppose you also wanted to use the triangle module in libigl. Then you would
change these to

```cmake
igl_include(glfw)
igl_include(restricted triangle)
…
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw igl_restricted::triangle)
```

The "restricted" appears in this case because the triangle library has a more
restricted license than libigl. See other examples commented out in
CMakeLists.txt.


## Dependencies

The only dependencies are STL, Eigen, [libigl](http://libigl.github.io/libigl/) and the dependencies
of the `igl::opengl::glfw::Viewer` (OpenGL, glad and GLFW).

The CMake build system will automatically download libigl and its dependencies using
[CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
thus requiring no setup on your part.