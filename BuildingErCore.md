# Introduction #

This page explains how to build exposure render with Visual Studio 2008. When everything goes well, CMake will generate a Visual Studio 2008 solution containing:
  * ErCore (Core Exposure Render libraries)
  * VtkEr (VTK wrapper for Exposure Render, option ER\_VTK in CMake)
  * VtkErExample (Example project that shows how to use the VtkEr, option ER\_VTK\_EXAMPLE in CMake)
  * vtkErPython (VTK Exposure Render python wrappings, option ER\_VTK\_PYTHON in CMake)
  * vtkErPythonD (VTK Exposure Render debug python wrappings, option ER\_VTK\_PYTHON in CMake)

Please note that this version of Exposure Render is designed to run on CUDA enabled hardware with streaming architecture of 2.0 and above. In case you want to compile for older streaming architectures you should follow the guidelines in CMakeLists.txt.

Build instructions:
  * Download + install a Mercurial client, I prefer this one: http://tortoisehg.bitbucket.org/download/.
  * Checkout the ercore repository: http://code.google.com/p/exposure-render/source/checkout?repo=ercore, use this revision 51632350aa48
  * Download and install the latest CMake: http://www.cmake.org/cmake/resources/software.html
  * Download and install the GPU Computing SDK, CUDA Toolkit and developer drivers from: http://developer.nvidia.com/cuda-downloads
  * For the VTK/Python wrapper: Download and compile the latest VTK: http://www.vtk.org/VTK/resources/software.html. My advise would be to compile in Debug and RelWithDebInfo mode.
  * Download and install Python 2.7: http://www.python.org/getit/
  * Run CMake on /ercore/Source
  * This will generate ExposureRender.sln in the CMake build dir
  * Compile the whole solution