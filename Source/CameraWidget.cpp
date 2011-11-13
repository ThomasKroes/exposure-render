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

#include "CameraWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"
#include "VtkWidget.h"

QCameraWidget::QCameraWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_FilmWidget(),
	m_ApertureWidget(),
	m_ProjectionWidget(),
	m_FocusWidget(),
	m_PresetsWidget(NULL, "Camera", "Camera")
{
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_PresetsWidget);
	m_MainLayout.addWidget(&m_FilmWidget);
	m_MainLayout.addWidget(&m_ApertureWidget);
	m_MainLayout.addWidget(&m_ProjectionWidget);
	m_MainLayout.addWidget(&m_FocusWidget);

	QObject::connect(&gCamera.GetFilm(), SIGNAL(Changed(const QFilm&)), &gCamera, SLOT(OnFilmChanged()));
	QObject::connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), &gCamera, SLOT(OnApertureChanged()));
	QObject::connect(&gCamera.GetProjection(), SIGNAL(Changed(const QProjection&)), &gCamera, SLOT(OnProjectionChanged()));
	QObject::connect(&gCamera.GetFocus(), SIGNAL(Changed(const QFocus&)), &gCamera, SLOT(OnFocusChanged()));
	QObject::connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	QObject::connect(&gStatus, SIGNAL(LoadPreset(const QString&)), &m_PresetsWidget, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&gCamera, SIGNAL(Changed()), this, SLOT(OnCameraChanged()));
}

void QCameraWidget::OnLoadPreset(const QString& Name)
{
	m_PresetsWidget.LoadPreset(gCamera, Name);

//	gScene.m_Camera.m_Target	= gCamera.GetTarget();
//	gScene.m_Camera.m_From		= gCamera.GetFrom();
//	gScene.m_Camera.m_Up		= gCamera.GetUp();
}

void QCameraWidget::OnSavePreset(const QString& Name)
{
	QCamera Preset(gCamera);
	Preset.SetName(Name);

//	Preset.SetFrom(gScene.m_Camera.m_From);
//	Preset.SetTarget(gScene.m_Camera.m_Target);
//	Preset.SetUp(gScene.m_Camera.m_Up);

	// Add the preset
	m_PresetsWidget.SavePreset(Preset);
}

QSize QCameraWidget::sizeHint() const
{
	return QSize(20, 20);
}

void QCameraWidget::OnCameraChanged(void)
{
	if (!gpActiveRenderWidget)
		return;

	gpActiveRenderWidget->GetCamera()->SetFocalDisk(gCamera.GetAperture().GetSize());
	gpActiveRenderWidget->GetCamera()->SetFocalDistance(gCamera.GetFocus().GetFocalDistance());
}