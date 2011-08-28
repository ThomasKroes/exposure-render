
#include "TransferFunctionWidget.h"
#include "TransferFunctionView.h"
#include "NodePropertiesWidget.h"
#include "GradientView.h"

QTransferFunctionWidget::QTransferFunctionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_pMainLayout(NULL),
	m_pTransferFunction(NULL),
	m_pTransferFunctionView(NULL),
	m_pGradientView(NULL),
	m_pNodePropertiesWidget(NULL)
{
	setTitle("Transfer Function");
	setToolTip("Transfer function properties");

	// Create main layout
	m_pMainLayout = new QGridLayout();
	m_pMainLayout->setAlignment(Qt::AlignTop);
	setLayout(m_pMainLayout);

	// Create Qt transfer function
	m_pTransferFunction = new QTransferFunction();

	// Transfer function view
	m_pTransferFunctionView = new QTransferFunctionView(this);
	m_pMainLayout->addWidget(m_pTransferFunctionView);

	// Gradient view
	m_pGradientView = new QGradientView(this);
	m_pMainLayout->addWidget(m_pGradientView);

	// Node properties
	m_pNodePropertiesWidget = new QNodePropertiesWidget(this);
	m_pMainLayout->addWidget(m_pNodePropertiesWidget);

	gTransferFunction.AddNode(new QNode(0.0f, 0.0f, QColor(255, 0, 0, 128), false));
	gTransferFunction.AddNode(new QNode(70.0f, 0.5f, QColor(255, 160, 30, 255)));
	gTransferFunction.AddNode(new QNode(100.0f, 0.1f, QColor(55, 160, 255, 255)));
	gTransferFunction.AddNode(new QNode(255.0f, 1.0f, QColor(10, 255, 0, 128), false));
}