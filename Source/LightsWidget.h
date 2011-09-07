#pragma once

#include <QtGui>

#include "Lighting.h"

class QLightItem : public QListWidgetItem
{
public:
	QLightItem(QListWidget* pListWidget, QLight* pLight) :
		QListWidgetItem(pListWidget),
		m_pLight(pLight)
	  {
		  setText(pLight->GetName());
	  }

	  QLight*	m_pLight;
};

class QLightsWidget : public QGroupBox
{
    Q_OBJECT

public:
    QLightsWidget(QWidget* pParent = NULL);

	virtual QSize sizeHint() const { return QSize(200, 200); }

protected slots:
	void UpdateLightList(void);
	void OnLightSelectionChanged(void);
	void OnLightSelectionChanged(QLight* pOldLight, QLight* pNewLight);
	void OnLightItemChanged(QListWidgetItem* pWidgetItem);
	void OnAddLight(void);
	void OnRemoveLight(void);
	void OnRenameLight(void);
	void OnCopyLight(void);
	void OnReset(void);

protected:
	QGridLayout		m_MainLayout;
	QListWidget		m_LightList;
	QPushButton		m_AddLight;
	QPushButton		m_RemoveLight;
	QPushButton		m_RenameLight;
	QPushButton		m_CopyLight;
	QPushButton		m_Reset;

	friend class QLightingWidget;
};