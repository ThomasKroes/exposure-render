/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "AboutDialog.h"

QAboutDialog::QAboutDialog(QWidget* pParent /*= NULL*/) :
	QDialog(pParent),
	m_MainLayout(),
	m_DialogButtons()
{
	setWindowIcon(GetIcon("question"));
	setContentsMargins(5, 5, 5, 5);
	
	setLayout(&m_MainLayout);

	QPushButton* pButton = NULL;

	pButton = new QPushButton();
	pButton->setIcon(GetIcon("about"));
	pButton->setIconSize(QSize(0.7 * 150, 0.7 * 178));
	pButton->setFlat(true);
	pButton->setStyleSheet("border: 0px");
	pButton->setEnabled(false);

	m_MainLayout.addWidget(pButton, 0, 0, Qt::AlignTop);

	QSettings Settings;

	QString AboutString;

	AboutString += "<b>About Exposure Render</b>";
	AboutString += "<p>This application accompanies the paper on : <b>Raytraced Lighting in Direct Volume Rendering (DVR)</b></p>";
	AboutString += "<p>Current version: " + Settings.value("version", "1.0.0").toString() + "</p>";
	AboutString += "<p>Exposure Render uses the following libraries/toolkits:</p>";
	AboutString += "<ul>";
		AboutString += "<b><li>VTK</b>The <a href='http://www.vtk.org/'>Visualization Toolkit</a> (VTK) is an open-source, freely available software system for 3D computer graphics, image processing and visualization</li><br>";
		AboutString += "<b><li>Fugue Icons</b>The <a href='http://code.google.com/p/fugue-icons-src/'>Fugue Icons</a> library comprises a comprehensive database of icons</li><br>";
		AboutString += "<b><li>Cuda</b><a href='http://www.nvidia.com/object/cuda_home_new.html'>CUDA™</a> is NVIDIA's parallel computing architecture. It enables dramatic increases in computing performance by harnessing the power of the GPU</li><br>";
		AboutString += "<b><li>Qt</b>The <a href='http://qt.nokia.com/products/'>Qt SDK</a> combines the Qt framework with tools designed to streamline the creation of applications for Symbian and Maemo, MeeGo (Nokia N9) in addition to desktop platforms, such as Microsoft Windows, Mac OS X, and Linux</li><br>";
	AboutString += "</ul>";
	AboutString += "<br>";
	AboutString += "<b>Acknowledgements</b><br>VolVis and Volume Library website for hosting data sets, J. E. Clarenburg <a href='http://www.clarenburg.nl/'>Studio Clarenburg</a> for the logo design";
	AboutString += "<br><br>";
	AboutString += "For more information please refer to the <a href='http://code.google.com/p/exposure-render/'>Exposure Render</a> website or to the <a href='http://graphics.tudelft.nl'>TU Delft Graphics Website</a>. You can also contact me directly: t.kroes@tudelft.nl";

	QLabel* pLabel = new QLabel(AboutString);
	
	pLabel->setWordWrap(true);
	pLabel->setStyleSheet("ul {list-style-type: none; background-image: url(navi_bg.png); height: 80px; width: 663px; margin: auto; }");

	m_MainLayout.addWidget(pLabel, 0, 1);
		
	m_DialogButtons.setStandardButtons(QDialogButtonBox::Ok);

	connect(&m_DialogButtons, SIGNAL(accepted()), this, SLOT(accept()));

	m_MainLayout.addWidget(&m_DialogButtons, 1, 0, 1, 2);
}