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

#include "BackgroundWidget.h"
#include "LightsWidget.h"
#include "RenderThread.h"

QBackgroundWidget::QBackgroundWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_GradientColorTopLabel(),
	m_GradientColorTop(),
	m_GradientColorMiddleLabel(),
	m_GradientColorMiddle(),
	m_GradientColorBottomLabel(),
	m_GradientColorBottom(),
	m_IntensityLabel(),
	m_IntensitySlider(),
	m_IntensitySpinner(),
	m_UseTexture(),
	m_TextureFilePath(),
	m_LoadTexture()
{
	// Title, status and tooltip
	setTitle("Background Illumination");
	setToolTip("Background Illumination");
	setStatusTip("Background Illumination");

	// Allow user to turn background illumination on/off
	setCheckable(true);

	// Apply main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Gradient color top
	m_GradientColorTopLabel.setText("Top");
	m_GradientColorTopLabel.setFixedWidth(50);
	m_MainLayout.addWidget(&m_GradientColorTopLabel, 0, 0);
	m_MainLayout.addWidget(&m_GradientColorTop, 0, 1, 1, 3);

	QObject::connect(&m_GradientColorTop, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorTopChanged(const QColor&)));

	// Gradient color Middle
	m_GradientColorMiddleLabel.setText("Middle");
	m_MainLayout.addWidget(&m_GradientColorMiddleLabel, 1, 0);
	m_MainLayout.addWidget(&m_GradientColorMiddle, 1, 1, 1, 3);

	QObject::connect(&m_GradientColorMiddle, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorMiddleChanged(const QColor&)));

	// Gradient color Bottom
	m_GradientColorBottomLabel.setText("Bottom");
	m_MainLayout.addWidget(&m_GradientColorBottomLabel, 2, 0);
	m_MainLayout.addWidget(&m_GradientColorBottom, 2, 1, 1, 3);

	QObject::connect(&m_GradientColorBottom, SIGNAL(currentColorChanged(const QColor&)), this, SLOT(OnGradientColorBottomChanged(const QColor&)));

	// Intensity
	m_IntensityLabel.setText("Intensity");
	m_MainLayout.addWidget(&m_IntensityLabel, 3, 0);

	m_IntensitySlider.setOrientation(Qt::Horizontal);
	m_IntensitySlider.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_IntensitySlider, 3, 1, 1, 2);

	m_IntensitySpinner.setRange(0.0, 10.0);
	m_MainLayout.addWidget(&m_IntensitySpinner, 3, 3);

	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), &m_IntensitySpinner, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySpinner, SIGNAL(valueChanged(double)), &m_IntensitySlider, SLOT(setValue(double)));
	QObject::connect(&m_IntensitySlider, SIGNAL(valueChanged(double)), this, SLOT(OnIntensityChanged(double)));

	// Use Texture
// 	m_UseTexture.setText("Use Texture");
// 	m_MainLayout.addWidget(&m_UseTexture, 4, 1);

	// Texture
// 	m_MainLayout.addWidget(new QLabel("Texture"), 5, 0);

	// Path
// 	m_TextureFilePath.setFixedHeight(22);
// 	m_MainLayout.addWidget(&m_TextureFilePath, 5, 1, 1, 2);

// 	m_LoadTexture.setIcon(GetIcon("folder-open-image"));
// 	m_LoadTexture.setFixedWidth(22);
// 	m_LoadTexture.setFixedHeight(22);
// 	m_MainLayout.addWidget(&m_LoadTexture, 5, 3);
// 
 	QObject::connect(this, SIGNAL(toggled(bool)), this, SLOT(OnBackgroundIlluminationChanged(bool)));
// 	QObject::connect(&m_UseTexture, SIGNAL(stateChanged(int)), this, SLOT(OnUseTextureChanged(int)));
// 	QObject::connect(&m_LoadTexture, SIGNAL(clicked()), this, SLOT(OnLoadTexture()));

 	QObject::connect(&gLighting.Background(), SIGNAL(Changed()), this, SLOT(OnBackgroundChanged()));

	OnBackgroundChanged();
}

void QBackgroundWidget::OnBackgroundIlluminationChanged(bool Checked)
{
	gLighting.Background().SetEnabled(Checked);
}

void QBackgroundWidget::OnGradientColorTopChanged(const QColor& Color)
{
	gLighting.Background().SetTopColor(Color);
}

void QBackgroundWidget::OnGradientColorMiddleChanged(const QColor& Color)
{
	gLighting.Background().SetMiddleColor(Color);
}

void QBackgroundWidget::OnGradientColorBottomChanged(const QColor& Color)
{
	gLighting.Background().SetBottomColor(Color);
}

void QBackgroundWidget::OnIntensityChanged(double Intensity)
{
	gLighting.Background().SetIntensity(Intensity);
}

void QBackgroundWidget::OnUseTextureChanged(int UseTexture)
{
	gLighting.Background().SetUseTexture(UseTexture);
}

void QBackgroundWidget::OnLoadTexture(void)
{
}

void QBackgroundWidget::OnBackgroundChanged(void)
{
	setChecked(gLighting.Background().GetEnabled());

	m_GradientColorTop.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());
	m_GradientColorMiddle.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());
	m_GradientColorBottom.setEnabled(gLighting.Background().GetEnabled() && !gLighting.Background().GetUseTexture());

	m_GradientColorTop.SetColor(gLighting.Background().GetTopColor());
	m_GradientColorMiddle.SetColor(gLighting.Background().GetMiddleColor());
	m_GradientColorBottom.SetColor(gLighting.Background().GetBottomColor());

	m_IntensitySlider.setValue((double)gLighting.Background().GetIntensity(), true);

	// Use texture
	m_TextureFilePath.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
	m_UseTexture.setChecked(gLighting.Background().GetUseTexture());

	// Use texture
	m_LoadTexture.setEnabled(gLighting.Background().GetEnabled() && gLighting.Background().GetUseTexture());
}
