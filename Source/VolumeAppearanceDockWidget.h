#pragma once

#include <QtGui>

class CVolumeAppearanceDockWidget;
class QTransferFunctionWidget;

class QVolumeAppearancePresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QVolumeAppearancePresetsWidget(QWidget* pParent = NULL);

private:
	void CreateActions(void);

protected:
	QGridLayout*	m_pGridLayout;
	QLabel*			m_pNameLabel;
	QComboBox*		m_pPresetNameComboBox;
	QPushButton*	m_pLoadPresetPushButton;
	QPushButton*	m_pSavePresetPushButton;
	QPushButton*	m_pRemovePresetPushButton;
	QPushButton*	m_pRenamePresetPushButton;

    QWidgetAction*	m_pLoadAction;
};

class CVolumeAppearanceWidget : public QWidget
{
    Q_OBJECT

public:
    CVolumeAppearanceWidget(QWidget* pParent = NULL);

protected:
	QVBoxLayout*						m_pMainLayout;
	QVolumeAppearancePresetsWidget*		m_pVolumeAppearancePresetsWidget;
	QTransferFunctionWidget*			m_pTransferFunctionWidget;
};

class CVolumeAppearanceDockWidget : public QDockWidget
{
    Q_OBJECT

public:
    CVolumeAppearanceDockWidget(QWidget* pParent = NULL);

protected:
	CVolumeAppearanceWidget*	m_pVolumeAppearanceWidget;
};