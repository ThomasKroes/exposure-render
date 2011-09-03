#pragma once

#include <QtGui>

#include "TransferFunctionWidget.h"
#include "PresetsWidget.h"
#include "NodePropertiesWidget.h"
#include "TransferFunction.h"

class QVolumeAppearanceDockWidget;

class QVolumeAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceWidget(QWidget* pParent = NULL);
	
public slots:
	void OnLoadPreset(const QString& Name);
	void OnSavePreset(const QString& Name);

protected:
	QGridLayout							m_MainLayout;
	QTransferFunctionWidget				m_TransferFunctionWidget;
	QNodePropertiesWidget				m_NodePropertiesWidget;
	QPresetsWidget<QTransferFunction>	m_PresetsWidget;
};

class QVolumeAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	QVolumeAppearanceWidget		m_VolumeAppearanceWidget;
};