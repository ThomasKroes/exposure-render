
// Precompiled headers
#include "Stable.h"

#include "CameraWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"

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

	// Connections
	QObject::connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&gRenderStatus, SIGNAL(LoadPreset(const QString&)), &m_PresetsWidget, SLOT(OnLoadPreset(const QString&)));
	QObject::connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	QObject::connect(&gCamera, SIGNAL(Changed()), this, SLOT(Update()));
}

void QCameraWidget::OnLoadPreset(const QString& Name)
{
	// Only load the preset when it exists
	if (!m_PresetsWidget.HasPreset(Name))
		return;

	gCamera = m_PresetsWidget.GetPreset(Name);
}

void QCameraWidget::OnSavePreset(const QString& Name)
{
	QCamera Preset(gCamera);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.SavePreset(Preset);
}

void QCameraWidget::Update(void)
{
	if (!Scene())
		return;

	Scene()->m_Camera.m_Aperture.m_Size	= gCamera.GetAperture().GetSize();

	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}