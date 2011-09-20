#pragma once

class QHistogram : public QObject
{
	Q_OBJECT

public:
    QHistogram(QObject* pParent = NULL);
	QHistogram::QHistogram(const QHistogram& Other);
	QHistogram& operator = (const QHistogram& Other);

	bool			GetEnabled(void) const;
	void			SetEnabled(const bool& Enabled);
	QList<int>&		GetBins(void);
	void			SetBins(const QList<int>& Bins);
	void			SetBins(const int* pBins, const int& NoBins);
	void			CreatePixMap(void);
	QPixmap*		GetPixMap(void);
	int				GetMax(void) const;
	void			SetMax(const int& Max);
	void			Reset(void);

signals:
	void HistogramChanged(void);

private:
	bool			m_Enabled;
	QList<int>		m_Bins;
	int				m_Max;
 	QPixmap*		m_pPixMap;
};

// Histogram singleton
extern QHistogram gHistogram;