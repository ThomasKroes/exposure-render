#pragma once

#include <QtGui>

#include "TransferFunctionWidget.h"
#include "PresetsWidget.h"
#include "NodePropertiesWidget.h"


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
	QPresetsWidget						m_PresetsWidget;
	QNodePropertiesWidget				m_NodePropertiesWidget;
	QTemplateWidget<QTransferFunction>	m_Presets;
};

class QVolumeAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	QVolumeAppearanceWidget		m_VolumeAppearanceWidget;
};