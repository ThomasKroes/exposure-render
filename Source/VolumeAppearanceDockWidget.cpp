
#include "VolumeAppearanceDockWidget.h"
#include "MainWindow.h"

QVolumeAppearanceWidget::QVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_MainLayout(),
	m_VolumeAppearancePresetsWidget(),
	m_TransferFunctionWidget(),
	m_NodePropertiesWidget()
{
	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TransferFunctionWidget, 0, 0);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 1, 0);
	m_MainLayout.addWidget(&m_VolumeAppearancePresetsWidget, 2, 0);
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