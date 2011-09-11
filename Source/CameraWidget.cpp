
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
	connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
	connect(&gRenderStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	connect(&gRenderStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));
}

void QCameraWidget::OnLoadPreset(const QString& Name)
{
	gCamera = m_PresetsWidget.GetPreset(Name);
}

void QCameraWidget::OnSavePreset(const QString& Name)
{
	QCamera Preset(gCamera);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.SavePreset(Preset);
}

void QCameraWidget::OnRenderBegin(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QCamera::Default());
	m_PresetsWidget.LoadPreset("Default");
}

void QCameraWidget::OnRenderEnd(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QCamera::Default());
	m_PresetsWidget.LoadPreset("Default");
}

void QCameraWidget::Update(void)
{
	// Flag the film resolution as dirty, this will restart the rendering
	//	Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}