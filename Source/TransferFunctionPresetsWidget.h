#pragma once

#include <QtGui>

class CVolumeAppearanceDockWidget;
class QTransferFunctionWidget;

class QTransferFunctionPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);

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