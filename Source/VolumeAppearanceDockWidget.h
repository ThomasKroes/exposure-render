#pragma once

#include <QtGui>

#include "TransferFunctionWidget.h"
#include "TransferFunctionPresetsWidget.h"
#include "NodePropertiesWidget.h"

class QVolumeAppearanceDockWidget;

class QVolumeAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceWidget(QWidget* pParent = NULL);

protected:
	QGridLayout							m_MainLayout;
	QTransferFunctionWidget				m_TransferFunctionWidget;
	QTransferFunctionPresetsWidget		m_VolumeAppearancePresetsWidget;
	QNodePropertiesWidget				m_NodePropertiesWidget;
};

class QVolumeAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	QVolumeAppearanceWidget		m_VolumeAppearanceWidget;
};