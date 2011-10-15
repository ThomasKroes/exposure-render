
// Precompiled headers
#include "Stable.h"

#include "HardwareInfoWidget.h"

QHardwareInfoWidget::QHardwareInfoWidget(QWidget* pParent) :
	QGroupBox(pParent),
	m_MainLayout()
{
	setTitle("Hardware");
	setStatusTip("Hardware");
	setToolTip("Hardware");

	m_MainLayout.setColumnMinimumWidth(0, 75);
	setLayout(&m_MainLayout);
}