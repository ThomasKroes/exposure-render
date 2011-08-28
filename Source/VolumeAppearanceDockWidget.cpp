
#include "VolumeAppearanceDockWidget.h"
#include "TransferFunctionWidget.h"
#include "MainWindow.h"

QVolumeAppearancePresetsWidget::QVolumeAppearancePresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pNameLabel(NULL),
	m_pPresetNameComboBox(NULL),
	m_pLoadPresetPushButton(NULL),
	m_pSavePresetPushButton(NULL),
	m_pRemovePresetPushButton(NULL),
	m_pRenamePresetPushButton(NULL),
	m_pLoadAction(NULL)
{
	setTitle("Presets");
	setToolTip("Transfer function presets");

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setColumnStretch(0, 1);
	m_pGridLayout->setColumnStretch(1, 1);
	m_pGridLayout->setColumnStretch(2, 1);
	m_pGridLayout->setColumnStretch(3, 1);
	m_pGridLayout->setColumnStretch(4, 1);
	m_pGridLayout->setColumnStretch(5, 1);
	m_pGridLayout->setAlignment(Qt::AlignTop);

	setLayout(m_pGridLayout);

	QSizePolicy SizePolicy;
	SizePolicy.setHorizontalPolicy(QSizePolicy::Fixed);
	SizePolicy.setVerticalPolicy(QSizePolicy::Maximum);
	SizePolicy.setHorizontalStretch(5);
	SizePolicy.setVerticalStretch(0);

	// Film width
	m_pNameLabel = new QLabel("Name");
	m_pGridLayout->addWidget(m_pNameLabel, 0, 0);

	m_pPresetNameComboBox = new QComboBox(this);
	m_pPresetNameComboBox->addItem("Medical");
	m_pPresetNameComboBox->addItem("Engineering");
	m_pPresetNameComboBox->setEditable(true);
	m_pGridLayout->addWidget(m_pPresetNameComboBox, 0, 1);

	m_pLoadPresetPushButton = new QPushButton("");
	m_pLoadPresetPushButton->setFixedWidth(20);
	m_pLoadPresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pLoadPresetPushButton, 0, 2);

	m_pSavePresetPushButton = new QPushButton(">");
	m_pSavePresetPushButton->setFixedWidth(20);
	m_pSavePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pSavePresetPushButton, 0, 3);

	m_pRemovePresetPushButton = new QPushButton("-");
	m_pRemovePresetPushButton->setFixedWidth(20);
	m_pRemovePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pRemovePresetPushButton, 0, 4);

	m_pRenamePresetPushButton = new QPushButton(".");
	m_pRenamePresetPushButton->setFixedWidth(20);
	m_pRenamePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pRenamePresetPushButton, 0, 5);
}

void QVolumeAppearancePresetsWidget::CreateActions(void)
{
	m_pLoadAction = new QWidgetAction(this);
    m_pLoadAction->setStatusTip(tr("Load an existing transfer function"));
	m_pLoadAction->setToolTip(tr("Load an existing transfer function"));
	connect(m_pLoadAction, SIGNAL(triggered()), this, SLOT(Open()));
	m_pLoadPresetPushButton->addAction(m_pLoadAction);
	gpMainWindow->m_pFileMenu->addAction(m_pLoadAction);

}

CVolumeAppearanceWidget::CVolumeAppearanceWidget(QWidget* pParent) :
	QWidget(pParent),
	m_pMainLayout(NULL),
	m_pVolumeAppearancePresetsWidget(NULL)
{
	// Create vertical layout
	m_pMainLayout = new QVBoxLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Volume appearance presets widget
	m_pVolumeAppearancePresetsWidget = new QVolumeAppearancePresetsWidget(this);
	m_pMainLayout->addWidget(m_pVolumeAppearancePresetsWidget);

	// Transfer function widget
	m_pTransferFunctionWidget = new QTransferFunctionWidget(this);
	m_pMainLayout->addWidget(m_pTransferFunctionWidget);
	
}

CVolumeAppearanceDockWidget::CVolumeAppearanceDockWidget(QWidget *parent) :
	QDockWidget(parent),
	m_pVolumeAppearanceWidget(NULL)
{
	setWindowTitle("Appearance");
	setToolTip("Volume Appearance");

	m_pVolumeAppearanceWidget = new CVolumeAppearanceWidget(this);

	setWidget(m_pVolumeAppearanceWidget);
}