
#Exposure Render

Exposure Render is a CUDA based volume raycaster, enhanced with physically based light transport. The implementation details are described in:

[An interactive photo-realistic volume rendering framework, T. Kroes, F. H. Post, C. P. Botha](http://graphics.tudelft.nl/Publications/kroes_exposure_2012)

![https://graphics.tudelft.nl/publications/](/Images/er_flyer_thumbnail.png)

##Building Exposure Render from source code
If you are eager to build Exposure Render yourself you should clone https://github.com/ThomasKroes/exposure-render.release110.git and follow [this](https://github.com/ThomasKroes/exposure-render.release110/blob/master/build.md) link.

For questions or comments contact me at t.kroes at lumc.nl/tudelft.nl.

##System requirements

* Microsoft Windows XP, Vista, or 7.
* At least 1GB of system memory.
* NVIDIA CUDA-compatible GPU with compute capability 1.0 and at least 512 megabytes of DRAM. GTX270 or higher is recommended
* At the moment, larger data sets might give problems, we are working on that!

##Developer(s)

Thomas Kroes

Affiliations:

**Delft University of Technology (TU Delft)**  
Computer Graphics and Visualization (CGV)  
*t.kroes at tudelft.nl*

**Leids Universitair medisch centrum (LUMC)**  
Laboratorium voor Klinische en Experimentele Beeldverwerking (LKEB)  
*t.kroes at lumc.nl*

##Acknowledgements

* Osirix Imaging Software for sharing the medical data sets
* Volvis website for the engine and bonsai data set
* Fugue icons for the icon database
* Francois Malan for pointing out the fruit MRI data sets

##Tested system configurations

Exposure Render has been tested on the following system configurations using Nvidia hardware:

* Windows 7 (64 bit) + Quadro FX1700
* Windows 7 (64 bit) + GTS240
* Windows 7 (64 bit) + GTS250
* Windows 7 (64 bit) + GTS450
* Windows 7 (64 bit) + GTX260
* Windows 7 (64 bit) + GTX270
* Windows 7 (64 bit) + GTX460
* Windows 7 (64 bit) + GTX470
* Windows 7 (64 bit) + GTX560
* Windows 7 (64 bit) + GTX570
* Windows 7 (64 bit) + GTX580

*Please mention your complete system setup when you report a bug: (OS (32/64 bit), graphics card, driver version etc.), possibly along with screen shots and error messages. Help make Exposure Render stable!*
