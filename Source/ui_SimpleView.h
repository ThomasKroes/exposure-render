/********************************************************************************
** Form generated from reading UI file 'SimpleView.ui'
**
** Created: Fri Aug 19 17:26:27 2011
**      by: Qt User Interface Compiler version 4.7.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMPLEVIEW_H
#define UI_SIMPLEVIEW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QFrame>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QSplitter>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "QVTKWidget.h"

QT_BEGIN_NAMESPACE

class Ui_SimpleView
{
public:
    QAction *actionOpenFile;
    QAction *actionExit;
    QAction *actionPrint;
    QAction *actionHelp;
    QAction *actionSave;
    QWidget *centralwidget;
    QGroupBox *groupBoxGraph;
    QVBoxLayout *vboxLayout;
    QSplitter *splitter;
    QFrame *tableFrame;
    QVBoxLayout *vboxLayout1;
    QVTKWidget *qvtkWidget;
    QMenuBar *menubar;
    QMenu *menuFile;
    QStatusBar *statusbar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *SimpleView)
    {
        if (SimpleView->objectName().isEmpty())
            SimpleView->setObjectName(QString::fromUtf8("SimpleView"));
        SimpleView->resize(1002, 666);
        const QIcon icon = QIcon(QString::fromUtf8(":/Icons/help.png"));
        SimpleView->setWindowIcon(icon);
        SimpleView->setIconSize(QSize(22, 22));
        actionOpenFile = new QAction(SimpleView);
        actionOpenFile->setObjectName(QString::fromUtf8("actionOpenFile"));
        actionOpenFile->setEnabled(true);
        const QIcon icon1 = QIcon(QString::fromUtf8(":/Icons/fileopen.png"));
        actionOpenFile->setIcon(icon1);
        actionExit = new QAction(SimpleView);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        actionPrint = new QAction(SimpleView);
        actionPrint->setObjectName(QString::fromUtf8("actionPrint"));
        const QIcon icon2 = QIcon(QString::fromUtf8(":/Icons/print.png"));
        actionPrint->setIcon(icon2);
        actionHelp = new QAction(SimpleView);
        actionHelp->setObjectName(QString::fromUtf8("actionHelp"));
        actionHelp->setIcon(icon);
        actionSave = new QAction(SimpleView);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        const QIcon icon3 = QIcon(QString::fromUtf8(":/Icons/filesave.png"));
        actionSave->setIcon(icon3);
        centralwidget = new QWidget(SimpleView);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        groupBoxGraph = new QGroupBox(centralwidget);
        groupBoxGraph->setObjectName(QString::fromUtf8("groupBoxGraph"));
        groupBoxGraph->setGeometry(QRect(9, 9, 984, 569));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(groupBoxGraph->sizePolicy().hasHeightForWidth());
        groupBoxGraph->setSizePolicy(sizePolicy);
        vboxLayout = new QVBoxLayout(groupBoxGraph);
        vboxLayout->setSpacing(0);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        vboxLayout->setContentsMargins(0, 0, 0, 0);
        splitter = new QSplitter(groupBoxGraph);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        tableFrame = new QFrame(splitter);
        tableFrame->setObjectName(QString::fromUtf8("tableFrame"));
        vboxLayout1 = new QVBoxLayout(tableFrame);
        vboxLayout1->setSpacing(0);
        vboxLayout1->setObjectName(QString::fromUtf8("vboxLayout1"));
        vboxLayout1->setContentsMargins(0, 0, 0, 0);
        splitter->addWidget(tableFrame);
        qvtkWidget = new QVTKWidget(splitter);
        qvtkWidget->setObjectName(QString::fromUtf8("qvtkWidget"));
        sizePolicy.setHeightForWidth(qvtkWidget->sizePolicy().hasHeightForWidth());
        qvtkWidget->setSizePolicy(sizePolicy);
        qvtkWidget->setMinimumSize(QSize(300, 300));
        splitter->addWidget(qvtkWidget);

        vboxLayout->addWidget(splitter);

        SimpleView->setCentralWidget(centralwidget);
        menubar = new QMenuBar(SimpleView);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1002, 25));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        SimpleView->setMenuBar(menubar);
        statusbar = new QStatusBar(SimpleView);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        SimpleView->setStatusBar(statusbar);
        toolBar = new QToolBar(SimpleView);
        toolBar->setObjectName(QString::fromUtf8("toolBar"));
        toolBar->setOrientation(Qt::Horizontal);
        toolBar->setIconSize(QSize(22, 22));
        SimpleView->addToolBar(Qt::TopToolBarArea, toolBar);

        menubar->addAction(menuFile->menuAction());
        menuFile->addAction(actionOpenFile);
        menuFile->addAction(actionSave);
        menuFile->addSeparator();
        menuFile->addAction(actionPrint);
        menuFile->addAction(actionHelp);
        menuFile->addAction(actionExit);
        toolBar->addAction(actionOpenFile);
        toolBar->addAction(actionSave);
        toolBar->addSeparator();
        toolBar->addAction(actionPrint);
        toolBar->addAction(actionHelp);

        retranslateUi(SimpleView);

        QMetaObject::connectSlotsByName(SimpleView);
    } // setupUi

    void retranslateUi(QMainWindow *SimpleView)
    {
        SimpleView->setWindowTitle(QApplication::translate("SimpleView", "SimpleView", 0, QApplication::UnicodeUTF8));
        actionOpenFile->setText(QApplication::translate("SimpleView", "Open File...", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("SimpleView", "Exit", 0, QApplication::UnicodeUTF8));
        actionPrint->setText(QApplication::translate("SimpleView", "Print", 0, QApplication::UnicodeUTF8));
        actionHelp->setText(QApplication::translate("SimpleView", "Help", 0, QApplication::UnicodeUTF8));
        actionSave->setText(QApplication::translate("SimpleView", "Save", 0, QApplication::UnicodeUTF8));
        groupBoxGraph->setTitle(QApplication::translate("SimpleView", "Views", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("SimpleView", "File", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SimpleView: public Ui_SimpleView {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMPLEVIEW_H
