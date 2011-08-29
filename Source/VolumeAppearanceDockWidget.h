#pragma once

#include <QtGui>

class QVolumeAppearanceDockWidget;
class QTransferFunctionWidget;
class QTransferFunctionPresetsWidget;
class QNodePropertiesWidget;

class QVolumeAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceWidget(QWidget* pParent = NULL);

protected:
	QGridLayout*						m_pMainLayout;
	QTransferFunctionWidget*			m_pTransferFunctionWidget;
	QTransferFunctionPresetsWidget*		m_pVolumeAppearancePresetsWidget;
	QNodePropertiesWidget*				m_pNodePropertiesWidget;
};

class QVolumeAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    QVolumeAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	QVolumeAppearanceWidget*	m_pVolumeAppearanceWidget;
};