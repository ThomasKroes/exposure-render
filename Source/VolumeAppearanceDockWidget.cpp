
#include "VolumeAppearanceDockWidget.h"
#include "MainWindow.h"

QVolumeAppearanceWidget::QVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_TransferFunctionWidget(),
	m_NodePropertiesWidget(),
	m_PresetsWidget(NULL, "Appearance", "Appearance")
{
	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TransferFunctionWidget, 0, 0);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 1, 0);
	m_MainLayout.addWidget(&m_PresetsWidget);

	// Connections
	connect(&m_PresetsWidget, SIGNAL(LoadPreset(const QString&)), this, SLOT(OnLoadPreset(const QString&)));
	connect(&m_PresetsWidget, SIGNAL(SavePreset(const QString&)), this, SLOT(OnSavePreset(const QString&)));
}

void QVolumeAppearanceWidget::OnLoadPreset(const QString& Name)
{
	gTransferFunction = m_PresetsWidget.GetPreset(Name);
}

void QVolumeAppearanceWidget::OnSavePreset(const QString& Name)
{
	QTransferFunction Preset(gTransferFunction);
	Preset.SetName(Name);

	// Add the preset
	m_PresetsWidget.AddPreset(Preset);
}

QVolumeAppearanceDockWidget::QVolumeAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_VolumeAppearanceWidget()
{
	setWindowTitle("Appearance");
	setToolTip("<img src=':/Images/palette.png'><div>Volume Appearance</div>");
	setWindowIcon(QIcon(":/Images/palette.png"));

	m_VolumeAppearanceWidget.setParent(this);

	setWidget(&m_VolumeAppearanceWidget);
}
