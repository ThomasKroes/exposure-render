
// Precompiled headers
#include "Stable.h"

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

	setWindowIcon(GetIcon("folder-open-image"));

	// Create main layout
	m_MainLayout.setAlignment(Qt::AlignTop);
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(&m_TransferFunctionView);
}