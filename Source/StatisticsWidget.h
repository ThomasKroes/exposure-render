#pragma once

#include <QtGui>

class QStatisticsWidget : public QTreeWidget
{
    Q_OBJECT

public:
    QStatisticsWidget(QWidget* pParent = NULL);

	QSize sizeHint() const;

	void Init(void);
	void ExpandAll(const bool& Expand);

private:
	void PopulateTree(void);
	QTreeWidgetItem* AddItem(QTreeWidgetItem* pParent, const QString& Property, const QString& Value = "", const QString& Unit = "", const QString& Icon = "");
	void UpdateStatistic(const QString& Group, const QString& Name, const QString& Value, const QString& Unit, const QString& Icon = "");
	QTreeWidgetItem* FindItem(const QString& Name);
	void RemoveChildren(const QString& Name);

public slots:
	void OnRenderBegin(void);
	void OnRenderEnd(void);
	void OnPreRenderFrame(void);
	void OnPostRenderFrame(void);
	void OnStatisticChanged(const QString& Group, const QString& Name, const QString& Value, const QString& Unit, const QString& Icon = "");
	
private:
	QGridLayout		m_MainLayout;
};