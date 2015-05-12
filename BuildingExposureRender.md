# Introduction #

This page explains how to build exposure render with Visual Studio

# Dependencies #

Exposure render depends on: CUDA, Qt and VTK

# CUDA #
You can skip this step if you already have the CUDA SDK and toolkit installed

The encessary installers can be downloaded from:  http://developer.nvidia.com/cuda-toolkit-40
  * Download and install the GPU Computing SDK
  * Download and install the CUDA Toolkit
  * Download and install the developer drivers
# Qt #
Skip this step if you already have compiled Qt with Visual Studio 20xx

Download the Qt sources from: http://get.qt.nokia.com/qt/source/qt-everywhere-opensource-src-4.7.3.zip

Extract to a directory on your hard drive, e.g. C:\Qt\4.7.3. Note that the path may not contain spaces.

  * Add an environment variable called QTDIR and set it to the dir. created in the previous step
  * Open the Visual Studio command prompt and navigate towards the Qt dir.
  * Run the following command: _configure -debug-and-release -opensource -shared -no-qt3support -qt-sql-sqlite -phonon -phonon-backend -no-webkit -no-script -platform win32-msvc2010_
  * Press y to accept the license
  * Run nmake to build Qt

More references on building Qt:
http://thomasstockx.blogspot.com/2011/03/qt-472-in-visual-studio-2010.html
# VTK #
VTK must be compiled with Qt support
  * Download the latest sources from the VTK website: http://www.vtk.org/VTK/resources/software.html
  * Extract the zip file (for instance c:\VTK\5.61)
  * Run cmake and set the source dir. to the VTK root dir
  * Make sure to check the Qt check box in the VTK settings
  * For the GUI to run you need to build VTK with GUI support
  * You might also consider to build the examples
  * Generate the Visual Studio 2010 solution files

Open the generated solution and build it, RelWithDebugInfo is advised

Afetr building, copy the presets directory from the source dir to the build dir.