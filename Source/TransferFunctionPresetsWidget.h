#pragma once

#include <QtGui>
#include <QtXml\qdom.h>

#include "TransferFunction.h"

class QPresetItem : public QListWidgetItem
{
public:
	QPresetItem(QListWidget* pListWidget, QTransferFunction* pTransferFunction) :
		QListWidgetItem(pListWidget),
		m_pTransferFunction(pTransferFunction)
	{
		setText(pTransferFunction->GetName());
	}

	QTransferFunction*	m_pTransferFunction;
};


class QTransferFunctionPresetsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QTransferFunctionPresetsWidget(QWidget* pParent = NULL);
	~QTransferFunctionPresetsWidget(void);

	virtual QSize sizeHint() const { return QSize(250, 300); }

protected slots:
	void CreateConnections(void);
	void CreateUI(void);
	void LoadPresetsFromFile(const bool& ChoosePath = false);
	void SavePresetsToFile(const bool& ChoosePath = false);
	void UpdatePresetsList(void);
	void OnPresetSelectionChanged(void);
	void OnLoadPreset(void);
	void OnRemovePreset(void);
	void OnSavePreset(void);
	void OnLoadPresets(void);
	void OnSavePresets(void);
	void OnDummy(void);
	void OnPresetNameChanged(const QString& Text);
	void OnPresetItemChanged(QListWidgetItem* pWidgetItem);

protected:
	QGridLayout				m_MainLayout;
	QLineEdit				m_PresetName;
	QPushButton				m_LoadPreset;
	QPushButton				m_SavePreset;
	QPushButton				m_RemovePreset;
	QPushButton				m_LoadPresets;
	QPushButton				m_SavePresets;
	QPushButton				m_Dummy;
	QListWidget				m_PresetList;
	QTransferFunctionList	m_TransferFunctions;
};