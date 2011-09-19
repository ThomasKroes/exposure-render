
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

	QObject::connect(&gCamera.GetFilm(), SIGNAL(Changed(const QFilm&)), &gCamera, SLOT(OnFilmChanged()));
	QObject::connect(&gCamera.GetAperture(), SIGNAL(Changed(const QAperture&)), &gCamera, SLOT(OnApertureChanged()));
	QObject::connect(&gCamera.GetProjection(), SIGNAL(Changed(const QProjection&)), &gCamera, SLOT(OnProjectionChanged()));
	QObject::connect(&gCamera.GetFocus(), SIGNAL(Changed(const QFocus&)), &gCamera, SLOT(OnFocusChanged()));
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

	// Film
	Scene()->m_Camera.m_Film.m_Exposure = gCamera.GetFilm().GetExposure();

	if (gCamera.GetFilm().IsDirty())
	{
// 		Scene()->m_Camera.m_Film.m_Resolution.SetResX(gCamera.GetFilm().GetWidth());
// 
// 		Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

	// Aperture
	Scene()->m_Camera.m_Aperture.m_Size	= gCamera.GetAperture().GetSize();
	
	// Projection
	Scene()->m_Camera.m_FovV = gCamera.GetProjection().GetFieldOfView();

	// Focus
	Scene()->m_Camera.m_Focus.m_FocalDistance = gCamera.GetFocus().GetFocalDistance();

	Scene()->m_DirtyFlags.SetFlag(CameraDirty);
}