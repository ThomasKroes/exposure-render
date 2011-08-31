#pragma once

#include <QtGui>

class QLightingPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QLightingPresetsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(200, 200); }

private slots:
protected:
	QGridLayout		m_MainLayout;
	QListWidget		m_PresetList;
	QLineEdit		m_PresetName;
	QPushButton		m_LoadPreset;
	QPushButton		m_SavePreset;
	QPushButton		m_RemovePreset;
	QPushButton		m_LoadPresets;
	QPushButton		m_SavePresets;
};