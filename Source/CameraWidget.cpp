
#include "CameraWidget.h"
#include "MainWindow.h"
#include "Scene.h"

QCamera gCamera;

void QCamera::ReadXML(QDomElement& Parent)
{
	QPresetXML::ReadXML(Parent);
}

QDomElement QCamera::WriteXML(QDomDocument& DOM, QDomElement& Parent)
{
	// Create transfer function preset root element
	QDomElement Preset = QPresetXML::WriteXML(DOM, Parent);

	return Preset;
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

	m_MainLayout.addWidget(&m_FilmWidget);
	m_MainLayout.addWidget(&m_ApertureWidget);
	m_MainLayout.addWidget(&m_ProjectionWidget);
	m_MainLayout.addWidget(&m_FocusWidget);
	m_MainLayout.addWidget(&m_PresetsWidget);

	// Connections
	connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
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