
#include "VolumeAppearanceDockWidget.h"
#include "TransferFunctionPresetsWidget.h"
#include "TransferFunctionWidget.h"
#include "NodePropertiesWidget.h"
#include "MainWindow.h"

QVolumeAppearanceWidget::QVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pVolumeAppearancePresetsWidget(NULL),
	m_pTransferFunctionWidget(NULL),
	m_pNodePropertiesWidget(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Transfer function widget
	m_pTransferFunctionWidget = new QTransferFunctionWidget(this);
	m_pMainLayout->addWidget(m_pTransferFunctionWidget, 0, 0, 1, 2);
	
	// Volume appearance presets widget
	m_pVolumeAppearancePresetsWidget = new QTransferFunctionPresetsWidget(this);
	m_pMainLayout->addWidget(m_pVolumeAppearancePresetsWidget, 1, 0);

	// Node properties widget
	m_pNodePropertiesWidget = new QNodePropertiesWidget(this);
	m_pMainLayout->addWidget(m_pNodePropertiesWidget, 1, 1);
}

QVolumeAppearanceDockWidget::QVolumeAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_pVolumeAppearanceWidget(NULL)
{
	setWindowTitle("Appearance");
	setToolTip("Volume Appearance");

	m_pVolumeAppearanceWidget = new QVolumeAppearanceWidget(this);

	setWidget(m_pVolumeAppearanceWidget);
}