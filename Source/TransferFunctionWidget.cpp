
#include "TransferFunctionWidget.h"
#include "TransferFunction.h"

QTransferFunctionWidget::QTransferFunctionWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_TransferFunctionView()
{
	// Set the size policy, making sure the widget fits nicely in the layout
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

	// Title, status and tooltip
	setTitle("Transfer Function");
	setToolTip("Transfer function properties");
	setStatusTip("Transfer function properties");

	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	// Transfer function view
	m_TransferFunctionView.setParent(this);
	m_MainLayout.addWidget(&m_TransferFunctionView);

	gTransferFunction.AddNode(0.0f, 0.0f, QColor(255, 0, 0, 0));
	gTransferFunction.AddNode(255.0f, 1.0f, QColor(0, 255, 0, 255));
}