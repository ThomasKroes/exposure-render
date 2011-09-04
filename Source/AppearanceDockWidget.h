#pragma once

#include <QtGui>

#include "TransferFunctionWidget.h"
#include "PresetsWidget.h"
#include "NodePropertiesWidget.h"
#include "TransferFunction.h"

class QAppearanceDockWidget;

class QAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QAppearanceWidget(QWidget* pParent = NULL);
	
public slots:
	void OnLoadPreset(const QString& Name);
	void OnSavePreset(const QString& Name);

protected:
	QGridLayout							m_MainLayout;
	QTransferFunctionWidget				m_TransferFunctionWidget;
	QNodePropertiesWidget				m_NodePropertiesWidget;
	QPresetsWidget<QTransferFunction>	m_PresetsWidget;
};

class QAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	QAppearanceWidget		m_VolumeAppearanceWidget;
};