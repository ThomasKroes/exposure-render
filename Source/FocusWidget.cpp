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

#include "FocusWidget.h"
#include "Camera.h"

QFocusWidget::QFocusWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_GridLayout(),
	m_FocusTypeComboBox(),
	m_FocalDistanceSlider(),
	m_FocalDistanceSpinner()
{
	setTitle("Focus");
	setStatusTip("Focus properties");
	setToolTip("Focus properties");

	m_GridLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_GridLayout);

	// Focus type
	m_GridLayout.addWidget(new QLabel("Type"), 0, 0);

	m_FocusTypeComboBox.addItem("Automatic");
// 	m_FocusTypeComboBox.addItem("Pick");
 	m_FocusTypeComboBox.addItem("Manual");
	m_GridLayout.addWidget(&m_FocusTypeComboBox, 0, 1, 1, 2);
	
	connect(&m_FocusTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(SetFocusType(int)));

	// Focal distance
	m_GridLayout.addWidget(new QLabel("Focal distance"), 1, 0);

	m_FocalDistanceSlider.setOrientation(Qt::Horizontal);
    m_FocalDistanceSlider.setTickPosition(QDoubleSlider::NoTicks);
	m_FocalDistanceSlider.setRange(0.0, 5.0);
	m_GridLayout.addWidget(&m_FocalDistanceSlider, 1, 1);
	
    m_FocalDistanceSpinner.setRange(0.0, 5.0);
	m_FocalDistanceSpinner.setSuffix(" m");
	m_GridLayout.addWidget(&m_FocalDistanceSpinner, 1, 2);
	
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), &m_FocalDistanceSpinner, SLOT(setValue(double)));
	connect(&m_FocalDistanceSlider, SIGNAL(valueChanged(double)), this, SLOT(SetFocalDistance(double)));
	connect(&m_FocalDistanceSpinner, SIGNAL(valueChanged(double)), &m_FocalDistanceSlider, SLOT(setValue(double)));
	connect(&gCamera.GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged(const QFocus&)));

	gStatus.SetStatisticChanged("Camera", "Focus", "", "", "");
}

void QFocusWidget::SetFocusType(int FocusType)
{
	gCamera.GetFocus().SetType(FocusType);
}

void QFocusWidget::SetFocalDistance(const double& FocalDistance)
{
	gCamera.GetFocus().SetFocalDistance(FocalDistance);
}

void QFocusWidget::OnFocusChanged(const QFocus& Focus)
{
	m_FocalDistanceSlider.setValue(Focus.GetFocalDistance(), true);
	m_FocalDistanceSpinner.setValue(Focus.GetFocalDistance(), true);

	gStatus.SetStatisticChanged("Focus", "Focal Distance", QString::number(Focus.GetFocalDistance(), 'f', 2), "mm");
}