
#Exposure Render

This is the Open-source implementation of "Interactive direct volume rendering with physically-based lighting"

This is dedicated to Exposure Render, a CUDA based volume raycaster, enhanced with physically based light transport. The implementation details are described in the paper:

http://graphics.tudelft.nl/Publications/kroes_exposure_2012'>Exposure render: An interactive photo-realistic volume rendering framework, T. Kroes, F. H. Post, C. P. Botha, accepted at PLoS ONE

http://exposure-render.googlecode.com/hg/Images/er_flyer.pdf

![Alt text](/Images/er_flyer_thumbnail.png)

##Building Exposure Render from source code
If you are eager to build Exposure Render yourself you should clone the release 1.1.0http://code.google.com/p/exposure-render/source/clones'> mercurial repository, and not the default repository, because this repository is a WIP at the moment. Get more info on building Exposure Render with Visual Studio http://code.google.com/p/exposure-render/wiki/BuildingExposureRender'>here.
For questions or comments contact me at t.kroes at tudelft.nl.

You can also build the VTK wrapping and example project, find more info here http://code.google.com/p/exposure-render/wiki/BuildingErCore'>http://code.google.com/p/exposure-render/wiki/BuildingErCore. Note, this repo is still under development!


Join our https://groups.google.com/forum/#!forum/exposure-render'>Google Group to remain up-to-date on recent developments!


Overall demo of Exposure Render

Demonstration of hybrid scattering

Lighting

System requirements

Microsoft Windows XP, Vista, or 7.
At least 1GB of system memory.
NVIDIA CUDA-compatible GPU with compute capability 1.0 and at least 512 megabytes of DRAM. GTX270 or higher is recommended
At the moment, larger data sets might give problems, we are working on that!

Developer(s)

Thomas Kroes
PhD researcher at Delft University of Technology
http://exposure-render.googlecode.com/hg/Images/photo_thomas_kroes_thumbnail.png' />

Acknowledgements

Osirix Imaging Software for sharing the medical data sets
Volvis website for the engine and bonsai data set
Fugue icons for the icon database
Francois Malan for pointing out the fruit MRI data sets
Tested system configurations

Exposure Render has been tested on the following system configurations using Nvidia hardware:

Windows 7 (64 bit) + Quadro FX1700
Windows 7 (64 bit) + GTS240
Windows 7 (64 bit) + GTS250
Windows 7 (64 bit) + GTS450
Windows 7 (64 bit) + GTX260
Windows 7 (64 bit) + GTX270
Windows 7 (64 bit) + GTX460
Windows 7 (64 bit) + GTX470
Windows 7 (64 bit) + GTX560
Windows 7 (64 bit) + GTX570
Windows 7 (64 bit) + GTX580
Please mention your complete system setup when you http://code.google.com/p/exposure-render/issues/list'>submit a bug: (OS (32/64 bit), graphics card, driver version etc.), possibly along with screen shots and error messages. Help make Exposure Render stable!
