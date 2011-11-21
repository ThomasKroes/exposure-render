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

#include "ProjectionWidget.h"
#include "Camera.h"

QProjectionWidget::QProjectionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FieldOfViewSlider(),
	m_FieldOfViewSpinner()
{
	setTitle("Projection");
	setStatusTip("Projection properties");
	setToolTip("Projection properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Field of view
	m_GridLayout.addWidget(new QLabel("Field of view"), 4, 0);

	m_FieldOfViewSlider.setOrientation(Qt::Horizontal);
	m_FieldOfViewSlider.setRange(10.0, 150.0);
	m_GridLayout.addWidget(&m_FieldOfViewSlider, 4, 1);
	
    m_FieldOfViewSpinner.setRange(10.0, 150.0);
	m_FieldOfViewSpinner.setSuffix(" deg.");
	m_GridLayout.addWidget(&m_FieldOfViewSpinner, 4, 2);
	
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), &m_FieldOfViewSpinner, SLOT(setValue(double)));
	connect(&m_FieldOfViewSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFieldOfView(double)));
	connect(&m_FieldOfViewSpinner, SIGNAL(valueChanged(double)), &m_FieldOfViewSlider, SLOT(setValue(double)));
	connect(&gCamera.GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged(const QProjection&)));

	gStatus.SetStatisticChanged("Camera", "Projection", "", "", "");
}

void QProjectionWidget::SetFieldOfView(const double& FieldOfView)
{
	gCamera.GetProjection().SetFieldOfView(FieldOfView);
}

void QProjectionWidget::OnProjectionChanged(const QProjection& Projection)
{
	m_FieldOfViewSlider.setValue(Projection.GetFieldOfView(), true);
	m_FieldOfViewSpinner.setValue(Projection.GetFieldOfView(), true);

	gStatus.SetStatisticChanged("Projection", "Field Of View", QString::number(Projection.GetFieldOfView(), 'f', 2), "Deg.");
}