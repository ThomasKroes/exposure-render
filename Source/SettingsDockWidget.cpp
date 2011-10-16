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

#include "SettingsDockWidget.h"
#include "RenderThread.h"

CTracerSettingsWidget::CTracerSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_NoBouncesSlider(),
	m_NoBouncesSpinBox()
{
	setTitle("Tracer");
	setTitle("Tracer Properties");
	
	// Create grid layout
//	m_MainLayout.setColSpacing(0, 70);
	setLayout(&m_MainLayout);

	// Render type
	m_MainLayout.addWidget(new QLabel("Render type"), 0, 0);

	m_RenderTypeComboBox.addItem("Single scattering");
	m_RenderTypeComboBox.addItem("Multiple scattering");
	m_RenderTypeComboBox.addItem("MIP");
	m_MainLayout.addWidget(&m_RenderTypeComboBox, 0, 1, 1, 2);

	// No. bounces
	m_MainLayout.addWidget(new QLabel("No. bounces"), 1, 0);

	m_NoBouncesSlider.setOrientation(Qt::Horizontal);
    m_NoBouncesSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_NoBouncesSlider.setRange(0, 10);
	m_MainLayout.addWidget(&m_NoBouncesSlider, 1, 1);
	
    m_NoBouncesSpinBox.setRange(0, 10);
	m_MainLayout.addWidget(&m_NoBouncesSpinBox, 1, 2);
	
	connect(&m_NoBouncesSlider, SIGNAL(valueChanged(int)), &m_NoBouncesSpinBox, SLOT(setValue(int)));
	connect(&m_NoBouncesSpinBox, SIGNAL(valueChanged(int)), &m_NoBouncesSlider, SLOT(setValue(int)));

	// Phase
	m_MainLayout.addWidget(new QLabel("Scattering"), 3, 0);
}

CKernelSettingsWidget::CKernelSettingsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_KernelWidthSlider(),
	m_KernelWidthSpinBox(),
	m_KernelHeightSlider(),
	m_KernelHeightSpinBox()
{
	setTitle("Kernel");
	setToolTip("Kernel settings");

	// Create grid layout
//	m_MainLayout.setColSpacing(0, 70);
	setLayout(&m_MainLayout);

	// Kernel width
	m_MainLayout.addWidget(new QLabel("Kernel Width"), 3, 0);

	m_KernelWidthSlider.setOrientation(Qt::Horizontal);
    m_KernelWidthSlider.setFocusPolicy(Qt::StrongFocus);
    m_KernelWidthSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_KernelWidthSlider.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelWidthSlider, 3, 1);
	
    m_KernelWidthSpinBox.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelWidthSpinBox, 3, 2);
	
	connect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelWidthSpinBox, SLOT(setValue(int)));
	connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelWidthSlider, SLOT(setValue(int)));
	connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelWidth(int)));

	// Kernel height
	m_MainLayout.addWidget(new QLabel("Kernel Height"), 4, 0);

	m_KernelHeightSlider.setOrientation(Qt::Horizontal);
    m_KernelHeightSlider.setFocusPolicy(Qt::StrongFocus);
    m_KernelHeightSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_KernelHeightSlider.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelHeightSlider, 4, 1);
	
    m_KernelHeightSpinBox.setRange(2, 64);
	m_MainLayout.addWidget(&m_KernelHeightSpinBox, 4, 2);
	
	m_LockKernelHeightCheckBox.setText("Lock");

	m_MainLayout.addWidget(&m_LockKernelHeightCheckBox, 4, 3);

	connect(&m_KernelHeightSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));
	connect(&m_KernelHeightSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
	connect(&m_KernelHeightSpinBox, SIGNAL(valueChanged(int)), this, SLOT(SetKernelHeight(int)));
	connect(&m_LockKernelHeightCheckBox, SIGNAL(stateChanged(int)), this, SLOT(LockKernelHeight(int)));
}

void CKernelSettingsWidget::SetKernelWidth(const int& KernelWidth)
{
	// Flag the render params as dirty, this will restart the rendering
	gScene.m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void CKernelSettingsWidget::SetKernelHeight(const int& KernelHeight)
{
	// Flag the render params as dirty, this will restart the rendering
	gScene.m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void CKernelSettingsWidget::LockKernelHeight(const int& Lock)
{
	m_KernelHeightSlider.setEnabled(!Lock);
	m_KernelHeightSpinBox.setEnabled(!Lock);

	if (Lock)
	{
		connect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
		connect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));

		m_KernelHeightSlider.setValue(m_KernelWidthSlider.value());
	}
	else
	{
		disconnect(&m_KernelWidthSlider, SIGNAL(valueChanged(int)), &m_KernelHeightSlider, SLOT(setValue(int)));
		disconnect(&m_KernelWidthSpinBox, SIGNAL(valueChanged(int)), &m_KernelHeightSpinBox, SLOT(setValue(int)));
	}
}

CSettingsWidget::CSettingsWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_TracerSettingsWidget(),
	m_KernelSettingsWidget()
{
	// Create vertical layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TracerSettingsWidget);
//	m_MainLayout.addWidget(&m_KernelSettingsWidget);
}

QSettingsDockWidget::QSettingsDockWidget(QWidget* pParent) :
	QDockWidget(pParent),
	m_SettingsWidget()
{
	setWindowTitle("Tracer Settings");
	setToolTip("<img src=':/Images/gear.png'><div>Tracer Properties</div>");
	setWindowIcon(GetIcon("gear"));

	setWidget(&m_SettingsWidget);
};