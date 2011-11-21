/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the TU Delft nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "ApertureWidget.h"
#include "Camera.h"

QApertureWidget::QApertureWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_SizeSlider(),
	m_SizeSpinner()
{
	setTitle("Aperture");
	setStatusTip("Aperture properties");
	setToolTip("Aperture properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Aperture size
	m_GridLayout.addWidget(new QLabel("Size"), 3, 0);

	m_SizeSlider.setOrientation(Qt::Horizontal);
	m_SizeSlider.setRange(0.0, 0.1);
	m_GridLayout.addWidget(&m_SizeSlider, 3, 1);
	
    m_SizeSpinner.setRange(0.0, 0.1);
	m_SizeSpinner.setSuffix(" mm");
	m_SizeSpinner.setDecimals(3);
	m_GridLayout.addWidget(&m_SizeSpinner, 3, 2);
	
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), &m_SizeSpinner, SLOT(setValue(double)));
	connect(&m_SizeSpinner, SIGNAL(valueChanged(double)), &m_SizeSlider, SLOT(setValue(double)));
	connect(&m_SizeSlider, SIGNAL(valueChanged(double)), this, SLOT(SetAperture(double)));
	connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged(const QAperture&)));

	gStatus.SetStatisticChanged("Camera", "Aperture", "", "", "");
}

void QApertureWidget::SetAperture(const double& Aperture)
{
	gCamera.GetAperture().SetSize(Aperture);
}

void QApertureWidget::OnApertureChanged(const QAperture& Aperture)
{
 	m_SizeSlider.setValue(Aperture.GetSize(), true);
	m_SizeSpinner.setValue(Aperture.GetSize(), true);

	gStatus.SetStatisticChanged("Aperture", "Size", QString::number(Aperture.GetSize(), 'f', 3), "mm");
}