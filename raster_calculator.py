# -*- coding: utf-8 -*-
"""
Raster Calculator - Main Plugin Class
Part of MAS Raster Processing Tools
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar
from qgis.core import QgsApplication
from .raster_calculator_dialog import RasterCalculatorDialog


class RasterCalculator:
    """Main plugin class for Raster Calculator."""
    
    def __init__(self, iface):
        """Constructor.
        
        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialize locale
        locale = QSettings().value('locale/userLocale')
        if locale:
            locale = locale[0:2]
            locale_path = os.path.join(self.plugin_dir, 'i18n', f'raster_calculator_{locale}.qm')
            
            if os.path.exists(locale_path):
                self.translator = QTranslator()
                self.translator.load(locale_path)
                QCoreApplication.installTranslator(self.translator)
        
        self.actions = []
        self.menu = self.tr('MAS Raster Processing')
        
        # Initialize toolbar reference
        self.toolbar = None
        self.toolbar_name = 'MASRasterProcessingToolbar'
        
        # Store dialog and provider references
        self.dlg = None
        self.provider = None

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('RasterCalculator', message)

    def initGui(self):
        """Initialize the GUI."""
        # Use raster_calculator.png as icon
        icon_path = os.path.join(self.plugin_dir, 'raster_calculator.png')
        
        # Find or create the MAS Raster Processing toolbar
        self.toolbar = self.iface.mainWindow().findChild(QToolBar, self.toolbar_name)
        if not self.toolbar:
            self.toolbar = self.iface.addToolBar('MAS Raster Processing')
            self.toolbar.setObjectName(self.toolbar_name)
        
        # Create QIcon - use a default icon if raster_calculator.png doesn't exist
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
        else:
            # Create a simple default icon if icon is missing
            icon = QIcon.fromTheme('accessories-calculator')
            if icon.isNull():
                icon = self.iface.style().standardIcon(
                    self.iface.style().SP_ComputerIcon
                )
        
        self.action = QAction(
            icon,
            self.tr('Raster Calculator'),
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.action.setEnabled(True)
        self.action.setStatusTip(self.tr('Perform map algebra calculations on raster layers'))
        self.action.setWhatsThis(self.tr('Raster Calculator - Map algebra expression builder'))
        self.action.setObjectName('raster_calculator_action')
        
        # Add to Raster -> MAS Raster Processing menu
        self.iface.addPluginToRasterMenu(self.menu, self.action)
        
        # Add to toolbar
        self.toolbar.addAction(self.action)
        
        # Store action for cleanup
        self.actions.append(self.action)
        
        # Make sure toolbar is visible
        self.toolbar.setVisible(True)
        


    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        # Remove from menu
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.menu, action)
        
        # Remove actions from toolbar
        if self.toolbar:
            for action in self.actions:
                self.toolbar.removeAction(action)
            
            # Check if toolbar is empty and remove it if so
            if len(self.toolbar.actions()) == 0:
                self.iface.mainWindow().removeToolBar(self.toolbar)
                self.toolbar = None
        
        # Clean up actions
        for action in self.actions:
            action.deleteLater()
        self.actions.clear()

    def run(self):
        """Run method that opens the dialog."""
        if self.dlg is None:
            self.dlg = RasterCalculatorDialog(self.iface)
        
        # Refresh layers list
        self.dlg.load_project_rasters()
        
        # Show the dialog
        self.dlg.show()
        result = self.dlg.exec_()
        
        # Reset dialog for next use
        if result:
            self.dlg = None
