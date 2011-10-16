#pragma once

class QCudaDevice
{
public:
	QCudaDevice(void){};

	QCudaDevice& QCudaDevice::operator=(const QCudaDevice& Other)
	{
		m_ID				= Other.m_ID;
		m_Name				= Other.m_Name;
		m_Capability		= Other.m_Capability;
		m_GlobalMemory		= Other.m_GlobalMemory;
		m_MemoryClockRate	= Other.m_MemoryClockRate;
		m_NoMultiProcessors	= Other.m_NoMultiProcessors;
		m_GpuClockSpeed		= Other.m_GpuClockSpeed;
		m_RegistersPerBlock	= Other.m_RegistersPerBlock;

		return *this;
	}

	int			m_ID;
	QString		m_Name;
	QString		m_Capability;
	QString		m_GlobalMemory;
	QString		m_MemoryClockRate;
	QString		m_NoMultiProcessors;
	QString		m_GpuClockSpeed;
	QString		m_RegistersPerBlock;
};

typedef QList<QCudaDevice> CudaDevices;

class QCudaDevicesModel : public QAbstractTableModel
{
     Q_OBJECT

 public:
     QCudaDevicesModel(QObject* pParent = NULL);
     QCudaDevicesModel(CudaDevices Devices, QObject* pParent = NULL);

     int rowCount(const QModelIndex &parent) const;
     int columnCount(const QModelIndex &parent) const;
     QVariant data(const QModelIndex &index, int role) const;
     QVariant headerData(int section, Qt::Orientation orientation, int role) const;
     Qt::ItemFlags flags(const QModelIndex &index) const;
     bool setData(const QModelIndex &index, const QVariant &value, int role=Qt::EditRole);
     bool insertRows(int position, int rows, const QModelIndex &index=QModelIndex());
     bool removeRows(int position, int rows, const QModelIndex &index=QModelIndex());

     CudaDevices getList();

	 void EnumerateDevices(void);
	 void AddDevice(const QCudaDevice& Device);
	 bool GetDevice(const int& DeviceID, QCudaDevice& CudaDevice);

 private:
     CudaDevices listOfPairs;
 };

class QHardwareWidget : public QGroupBox
{
    Q_OBJECT

public:
    QHardwareWidget(QWidget* pParent = NULL);

private slots:
	void OnOptimalDevice(void);
	void OnSelection(const QModelIndex& ModelIndex);
	void OnCudaDeviceChanged(void);

private:
	QGridLayout			m_MainLayout;
	QCudaDevicesModel	m_Model;
	QTableView			m_Devices;
	QPushButton			m_OptimalDevice;
};