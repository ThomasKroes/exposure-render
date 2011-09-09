
#include "CameraWidget.h"
#include "MainWindow.h"
#include "RenderThread.h"

QCamera gCamera;

QCamera::QCamera(QObject* pParent /*= NULL*/) :
	QPresetXML(pParent),
	m_Film(),
	m_Aperture(),
	m_Projection(),
	m_Focus()
{
}

QCamera::QCamera(const QCamera& Other)
{
	*this = Other;
};

QCamera& QCamera::operator=(const QCamera& Other)
{
	QPresetXML::operator=(Other);

	m_Film			= Other.m_Film;
	m_Aperture		= Other.m_Aperture;
	m_Projection	= Other.m_Projection;
	m_Focus			= Other.m_Focus;

	emit Changed();

	return *this;
}

QFilm& QCamera::GetFilm(void)
{
	return m_Film;
}

void QCamera::SetFilm(const QFilm& Film)
{
	m_Film = Film;
}

QAperture& QCamera::GetAperture(void)
{
	return m_Aperture;
}

void QCamera::SetAperture(const QAperture& Aperture)
{
	m_Aperture = Aperture;
}

QProjection& QCamera::GetProjection(void)
{
	return m_Projection;
}

void QCamera::SetProjection(const QProjection& Projection)
{
	m_Projection = Projection;
}

QFocus& QCamera::GetFocus(void)
{
	return m_Focus;
}

void QCamera::SetFocus(const QFocus& Focus)
{
	m_Focus = Focus;
}

void QCamera::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);

	m_Film.ReadXML(Parent);
	m_Aperture.ReadXML(Parent);
	m_Projection.ReadXML(Parent);
	m_Focus.ReadXML(Parent);
}

QDomElement QCamera::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Camera
	QDomElement Camera = DOM.createElement("Camera");
	Parent.appendChild(Camera);

	QPresetXML::WriteXML(DOM, Camera);

	m_Film.WriteXML(DOM, Camera);
	m_Aperture.WriteXML(DOM, Camera);
	m_Projection.WriteXML(DOM, Camera);
	m_Focus.WriteXML(DOM, Camera);

	return Camera;
}

QCamera QCamera::Default(void)
{
	QCamera DefaultCamera;

	DefaultCamera.SetName("Default");

	return DefaultCamera;
}

CCameraWidget::CCameraWidget(QWidget* pParent) :
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

void CCameraWidget::OnLoadPreset(const QString& Name)
{
	gCamera = m_PresetsWidget.GetPreset(Name);
}

void CCameraWidget::OnSavePreset(const QString& Name)
{
	QCamera Preset(gCamera);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.SavePreset(Preset);
}

void CCameraWidget::OnRenderBegin(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QCamera::Default());
	m_PresetsWidget.LoadPreset("Default");
}

void CCameraWidget::OnRenderEnd(void)
{
	// Add a default transfer function and load it
	m_PresetsWidget.InsertPreset(0, QCamera::Default());
	m_PresetsWidget.LoadPreset("Default");
}

void CCameraWidget::Update(void)
{
	// Flag the film resolution as dirty, this will restart the rendering
//	Scene()->m_DirtyFlags.SetFlag(FilmResolutionDirty);
}

