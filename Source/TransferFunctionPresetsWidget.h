#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "TransferFunction.h"


class QTransferFunctionPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);
	~QTransferFunctionPresetsWidget(void);

private slots:
	void LoadPresetsFromFile(void);
	void SavePresetsToFile(void);
	void UpdatePresetsList(void);
	void OnLoadPreset(void);
	void OnRemovePreset(void);
	void OnRenamePreset(void);
	void OnSavePreset(void);
	void OnLoadPresets(void);
	void OnSavePresets(void);

protected:
	QGridLayout					m_GridLayout;
	QLineEdit					m_PresetName;
	QPushButton					m_LoadPreset;
	QPushButton					m_SavePreset;
	QPushButton					m_RemovePreset;
	QPushButton					m_RenamePreset;
	QPushButton					m_LoadPresets;
	QPushButton					m_SavePresets;
	QListWidget					m_PresetList;
	QList<QTransferFunction>	m_TransferFunctions;
};