#pragma once

#include <QtGui>

#include "TransferFunctionView.h"

class QTransferFunctionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(10, 300); }

protected:
	QGridLayout				m_MainLayout;
	QTransferFunctionView	m_TransferFunctionView;
};