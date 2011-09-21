#pragma once

class QNode;

class QNodeSelectionWidget : public QGroupBox
{
	Q_OBJECT

public:
	QNodeSelectionWidget(QWidget* pParent = NULL);

private slots:
	void OnNodeSelectionChanged(QNode* pNode);
	void OnNodeSelectionChanged(const int& Index);
	void OnFirstNode(void);
	void OnPreviousNode(void);
	void OnNextNode(void);
	void OnLastNode(void);
	void OnDeleteNode(void);

private:
	void SetupSelectionUI(void);

private:
	QGridLayout		m_MainLayout;
	QComboBox		m_NodeSelection;
	QPushButton		m_FirstNode;
	QPushButton		m_PreviousNode;
	QPushButton		m_NextNode;
	QPushButton		m_LastNode;
	QPushButton		m_DeleteNode;
};