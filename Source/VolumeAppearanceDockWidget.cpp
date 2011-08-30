
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

	// Transfer function widget
	m_TransferFunctionWidget.setParent(this);
	m_MainLayout.addWidget(&m_TransferFunctionWidget, 0, 0, 1, 2);
	
	// Volume appearance presets widget
	m_VolumeAppearancePresetsWidget.setParent(this);
	m_MainLayout.addWidget(&m_VolumeAppearancePresetsWidget, 1, 0);

	// Node properties widget
	m_NodePropertiesWidget.setParent(this);
	m_MainLayout.addWidget(&m_NodePropertiesWidget, 1, 1);
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