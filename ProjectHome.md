<h1>Exposure Render</h1>
This Google Code project website is dedicated to Exposure Render, a CUDA based volume raycaster, enhanced with physically based light transport. The implementation details are described in the paper:

<i><a href='http://graphics.tudelft.nl/Publications/kroes_exposure_2012'>Exposure render: An interactive photo-realistic volume rendering framework</a>, T. Kroes, F. H. Post, C. P. Botha, accepted at PLoS ONE</i>

<a href='http://exposure-render.googlecode.com/hg/Images/er_flyer.pdf'><img src='http://exposure-render.googlecode.com/hg/Images/er_flyer_thumbnail.png' alt='Exposure Render Flyer' /></a>

Exposure Render has been officially released! You can download it <a href='http://code.google.com/p/exposure-render/downloads/detail?name=ExposureRender-1.1.0-win32.exe&can=2&q=#makechanges'>here</a>. We have managed to solve the most critical and annoying bugs and are still working hard to make it even more stable.

<b>Building Exposure Render</b><br>
If you are eager to build Exposure Render yourself you should clone the release 1.1.0<a href='http://code.google.com/p/exposure-render/source/clones'> mercurial repository</a>, and not the default repository, because this repository is a WIP at the moment. Get more info on building Exposure Render with Visual Studio <a href='http://code.google.com/p/exposure-render/wiki/BuildingExposureRender'>here</a>.<br>
For questions or comments contact me at t.kroes at tudelft.nl.<br>
<br>
You can also build the VTK wrapping and example project, find more info here <a href='http://code.google.com/p/exposure-render/wiki/BuildingErCore'>http://code.google.com/p/exposure-render/wiki/BuildingErCore</a>. Note, this repo is still under development!<br>
<br>
<br>
Join our <a href='https://groups.google.com/forum/#!forum/exposure-render'>Google Group</a> to remain up-to-date on recent developments!<br>
<br>
<table cellpadding='3' cellspacing='0'>
<tr>
<td><h2>Overall demo of Exposure Render</h2></td>
<td><h2>Demonstration of hybrid scattering</h2></td>
<td><h2>Lighting</h2></td>
</tr>
<tr>
<td><a href='http://www.youtube.com/watch?feature=player_embedded&v=qzFv0draRG8' target='_blank'><img src='http://img.youtube.com/vi/qzFv0draRG8/0.jpg' width='425' height=344 /></a></td>
<td><a href='http://www.youtube.com/watch?feature=player_embedded&v=4D2HfJ5Cwqc' target='_blank'><img src='http://img.youtube.com/vi/4D2HfJ5Cwqc/0.jpg' width='425' height=344 /></a></td>
<td><a href='http://www.youtube.com/watch?feature=player_embedded&v=cZaPIEo6PPs' target='_blank'><img src='http://img.youtube.com/vi/cZaPIEo6PPs/0.jpg' width='425' height=344 /></a></td>
</tr>
</table>

<h2>System requirements</h2>
<ul>
<li>Microsoft Windows XP, Vista, or 7.</li>
<li>At least 1GB of system memory.</li>
<li>NVIDIA CUDA-compatible GPU with compute capability 1.0 and at least 512 megabytes of DRAM. GTX270 or higher is recommended</li>
</ul>
<i>At the moment, larger data sets might give problems, we are working on that!</i>
<h2>Developer(s)</h2>
Thomas Kroes<br>
PhD researcher at Delft University of Technology<br>
<img src='http://exposure-render.googlecode.com/hg/Images/photo_thomas_kroes_thumbnail.png' />

<h2>Acknowledgements</h2>
<ul><li><a href='http://pubimage.hcuge.ch:8080/'>Osirix Imaging Software</a> for sharing the medical data sets</li>
<li><a href='http://www.volvis.org/'>Volvis website</a> for the <a href='http://www.gris.uni-tuebingen.de/edu/areas/scivis/volren/datasets/data/engine.raw.gz'>engine</a> and <a href='http://www.gris.uni-tuebingen.de/edu/areas/scivis/volren/datasets/data/bonsai.raw.gz'>bonsai</a> data set</li>
<li><a href='http://code.google.com/p/fugue-icons-src/'>Fugue icons</a> for the icon database</li>
<li>Francois Malan for pointing out the fruit MRI data sets</li>
</ul>

<h2>Tested system configurations</h2>
Exposure Render has been tested on the following system configurations using Nvidia hardware:<br>
<ul>
<li>Windows 7 (64 bit) + Quadro FX1700</li>
<li>Windows 7 (64 bit) + GTS240</li>
<li>Windows 7 (64 bit) + GTS250</li>
<li>Windows 7 (64 bit) + GTS450</li>
<li>Windows 7 (64 bit) + GTX260</li>
<li>Windows 7 (64 bit) + GTX270</li>
<li>Windows 7 (64 bit) + GTX460</li>
<li>Windows 7 (64 bit) + GTX470</li>
<li>Windows 7 (64 bit) + GTX560</li>
<li>Windows 7 (64 bit) + GTX570</li>
<li>Windows 7 (64 bit) + GTX580</li>
</ul>

<i>
Please mention your complete system setup when you <a href='http://code.google.com/p/exposure-render/issues/list'>submit</a> a bug: (OS (32/64 bit), graphics card, driver version etc.), possibly along with screen shots and error messages. Help make Exposure Render stable!</i>

<table cellpadding='2' cellspacing='10'>
<tr>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_01.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_02.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_03.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_04.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_05.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_06.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_07.png' /></td>
<td><img src='http://exposure-render.googlecode.com/hg/Images/example_08.png' /></td>
</tr>
</table>