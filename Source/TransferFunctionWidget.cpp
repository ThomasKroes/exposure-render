
#include "TransferFunctionWidget.h"
#include "TransferFunctionView.h"

QTransferFunctionWidget::QTransferFunctionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pMainLayout(NULL),
	m_pTransferFunction(NULL),
	m_pTransferFunctionView(NULL)
{
	// Title, status and tooltip
	setTitle("Transfer Function");
	setToolTip("Transfer function properties");
	setStatusTip("Transfer function properties");

	// Create main layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Create Qt transfer function
	m_pTransferFunction = new QTransferFunction();

	// Transfer function view
	m_pTransferFunctionView = new QTransferFunctionView(this);
	m_pMainLayout->addWidget(m_pTransferFunctionView);

	gTransferFunction.AddNode(0.0f, 0.0f, QColor(255, 0, 0, 0));
	gTransferFunction.AddNode(255.0f, 1.0f, QColor(0, 255, 0, 255));
}