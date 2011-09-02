
#include "VolumeAppearanceDockWidget.h"
#include "MainWindow.h"

QVolumeAppearanceWidget::QVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_TransferFunctionWidget(),
	m_NodePropertiesWidget(),
	m_PresetsWidget("AppearancePresets.xml", this),
	m_Presets("Appearance")
{
	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TransferFunctionWidget, 0, 0);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 1, 0);
//	m_MainLayout.addWidget(&m_PresetsWidget, 2, 0);

	

	m_MainLayout.addWidget(&m_Presets);

	m_Presets.m_Presets.append(QTransferFunction());

	// Connections
	connect(&m_Presets, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_Presets, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
}

void QVolumeAppearanceWidget::OnLoadPreset(const QString& Name)
{
}

void QVolumeAppearanceWidget::OnSavePreset(const QString& Name)
{
}

QVolumeAppearanceDockWidget::QVolumeAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_VolumeAppearanceWidget()
{
	setWindowTitle("Appearance");
	setToolTip("Volume Appearance");

	m_VolumeAppearanceWidget.setParent(this);

	setWidget(&m_VolumeAppearanceWidget);
}
