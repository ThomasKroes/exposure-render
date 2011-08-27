#pragma once

#include <QtGui>

#include "TransferFunction.h"

class QTransferFunction;
class QTransferFunctionView;
class QGradientView;
class QNodeItem;
class QNodePropertiesWidget;

class QTransferFunctionWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionWidget(QWidget* pParent = NULL);

protected:
	QGridLayout*				m_pMainLayout;
	QTransferFunction*			m_pTransferFunction;
	QTransferFunctionView*		m_pTransferFunctionView;
	QGradientView*				m_pGradientView;
	QNodePropertiesWidget*		m_pNodePropertiesWidget;
};