
#include "TransferFunctionPresetsWidget.h"


QTransferFunctionPresetsModel::QTransferFunctionPresetsModel(QObject* pParent) :
	QAbstractListModel(pParent)
{
	beginInsertRows(QModelIndex(), 0, 0);

	QTransferFunction TfBonsai;
	TfBonsai.SetName("Bonsai");

	QTransferFunction TfEngine;
	TfEngine.SetName("Engine");

	m_TransferFunctions.append(TfBonsai);
	m_TransferFunctions.append(TfEngine);

	endInsertRows();
}

QVariant QTransferFunctionPresetsModel::data(const QModelIndex& Index, int Role) const
{
	if (!Index.isValid())
		return QVariant();

	if (Index.row() >= m_TransferFunctions.size())
		return QVariant();

	return QVariant(m_TransferFunctions[Index.row()].GetName());
/*
	switch (Role)
	{
		case Qt::DisplayPropertyRole: return m_TransferFunctions[Index.row()];
		case Qt::ToolTipRole: return "Name of the transfer function";
		case Qt::StatusTipRole: return "Name of the transfer function";
		case Qt::FontRole: return 5;
		case Qt::TextAlignmentRole: return 7;
		case Qt::BackgroundColorRole: return QColor(255, 0, 0);
		case Qt::TextColorRole: return QColor(255, 255, 255);
		case Qt::CheckStateRole: return true;
		

		default: return QVariant();
	}*/
}

QTransferFunctionPresetsWidget::QTransferFunctionPresetsWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pGridLayout(NULL),
	m_pNameLabel(NULL),
	m_pPresetNameEdit(NULL),
	m_pLoadPresetsPushButton(NULL),
	m_pSavePresetsPushButton(NULL),
	m_pSavePresetPushButton(NULL),
	m_pRemovePresetPushButton(NULL),
	m_Model(100, 1)
{
	// Title, status and tooltip
	setTitle("Transfer Function Presets");
	setToolTip("Transfer Function Presets");
	setStatusTip("Transfer Function Presets");

//	setFixedHeight(150);
	setFixedWidth(200);

	// Create grid layout
	m_pGridLayout = new QGridLayout();
	m_pGridLayout->setAlignment(Qt::AlignTop);

	setLayout(m_pGridLayout);

	// Film width
	m_pNameLabel = new QLabel("Name");
//	m_pGridLayout->addWidget(m_pNameLabel, 0, 0);

	m_pTable = new QListWidget();
	m_pTable->setCaption("asdasd");
	m_pTable->setAlternatingRowColors(true);
	m_pTable->addItem("Bonsai");
	m_pTable->addItem("Engine");
	m_pTable->addItem("Manix");

	m_pGridLayout->addWidget(m_pTable, 0, 0, 1, 5);

	m_pPresetNameEdit = new QLineEdit(this);
	m_pGridLayout->addWidget(m_pPresetNameEdit, 1, 0);

	m_pLoadPresetsPushButton = new QPushButton("L");
	m_pLoadPresetsPushButton->setToolTip("Load presets from files");
	m_pLoadPresetsPushButton->setStatusTip("Load transfer function presets from file");
	m_pLoadPresetsPushButton->setFixedWidth(20);
	m_pLoadPresetsPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pLoadPresetsPushButton, 1, 1);

	m_pSavePresetsPushButton = new QPushButton("S");
	m_pSavePresetsPushButton->setToolTip("Save presets to file");
	m_pSavePresetsPushButton->setStatusTip("Save transfer function presets to file");
	m_pSavePresetsPushButton->setFixedWidth(20);
	m_pSavePresetsPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pSavePresetsPushButton, 1, 2);

	m_pSavePresetPushButton = new QPushButton("S");
	m_pSavePresetPushButton->setToolTip("Save Preset");
	m_pSavePresetPushButton->setStatusTip("Save transfer function preset");
	m_pSavePresetPushButton->setFixedWidth(20);
	m_pSavePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pSavePresetPushButton, 1, 3);

	m_pRemovePresetPushButton = new QPushButton("R");
	m_pRemovePresetPushButton->setToolTip("Remove Preset");
	m_pRemovePresetPushButton->setStatusTip("Remove transfer function preset");
	m_pRemovePresetPushButton->setFixedWidth(20);
	m_pRemovePresetPushButton->setFixedHeight(20);
	m_pGridLayout->addWidget(m_pRemovePresetPushButton, 1, 4);
}