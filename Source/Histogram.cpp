
#include "Histogram.h"

QHistogram gHistogram;

QHistogram::QHistogram(QObject* pParent /*= NULL*/) :
	QObject(pParent),
	m_Enabled(false),
	m_Bins(),
	m_Max(0)
{
}

QHistogram::QHistogram(const QHistogram& Other)
{
	*this = Other;
};

QHistogram& QHistogram::operator=(const QHistogram& Other)
{
	m_Enabled	= Other.m_Enabled;
	m_Bins		= Other.m_Bins;
	m_Max		= Other.m_Max;

	return *this;
}

bool QHistogram::GetEnabled(void) const
{
	return m_Enabled;
}

void QHistogram::SetEnabled(const bool& Enabled)
{
	m_Enabled = Enabled;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

QList<int>& QHistogram::GetBins(void)
{
	return m_Bins;
}

void QHistogram::SetBins(const QList<int>& Bins)
{
	m_Bins = Bins;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

void QHistogram::SetBins(const int* pBins, const int& NoBins)
{
	// Clear the bin list
	m_Bins.clear();

	m_Max = 0;

	for (int i = 0; i < NoBins; i++)
	{
		if (pBins[i] > GetMax())
			m_Max = pBins[i];
	}

	for (int i = 0; i < NoBins; i++)
		m_Bins.append(pBins[i]);

	m_Enabled = true;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

int QHistogram::GetMax(void) const
{
	return m_Max;
}

void QHistogram::SetMax(const int& Max)
{
	m_Max = Max;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}

void QHistogram::Reset(void)
{
	m_Enabled = false;
	m_Bins.clear();
	m_Max = 0;

	// Inform others that the histogram has changed
	emit HistogramChanged();
}