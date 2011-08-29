
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
	m_pPresetNameComboBox(NULL),
	m_pLoadPresetPushButton(NULL),
	m_pSavePresetPushButton(NULL),
	m_pRemovePresetPushButton(NULL),
	m_pRenamePresetPushButton(NULL),
	m_Model(100, 1)
{
	// Title, status and tooltip
	setTitle("Transfer Function Presets");
	setToolTip("Transfer Function Presets");
	setStatusTip("Transfer Function Presets");

//	setFixedHeight(150);
	setFixedWidth(150);

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
	m_pTable->addItem("asd");
	m_pTable->addItem("asd");
	m_pTable->addItem("asd");
	m_pTable->addItem("asd");
	/*
	m_pTable->setColumnCount(2);
	m_pTable->setColumnWidth(0, 15);
	m_pTable->setColumnWidth(1, 15);
	
	m_pTable->horizontalHeader()->setResizeMode(0, QHeaderView::Stretch);
	m_pTable->horizontalHeader()->setResizeMode(1, QHeaderView::Stretch);
	

	m_pTable->verticalHeader()->hide();

//	m_pTable->setItem(0, 0, new QTableWidgetItem("Bonsai"));

	m_pTable->horizontalHeader()->setStretchLastSection(true);
	
//	m_Model.setParent(this);

	m_pTable->setModel(&m_Model);
	*/

	m_pGridLayout->addWidget(m_pTable, 0, 0);


	/*
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
	*/
}