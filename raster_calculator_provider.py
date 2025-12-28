# -*- coding: utf-8 -*-
"""
Raster Calculator Processing Provider
"""

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os
from .raster_calculator_algorithm import RasterCalculatorAlgorithm


class RasterCalculatorProvider(QgsProcessingProvider):
    """
    Processing Provider for Raster Calculator.
    Part of MAS Geospatial Tools.
    """
    
    def __init__(self):
        super().__init__()

    def loadAlgorithms(self):
        """Load all algorithms for this provider."""
        self.addAlgorithm(RasterCalculatorAlgorithm())

    def id(self):
        """Unique provider ID."""
        return 'raster_calculator'

    def name(self):
        """Provider name shown in Processing Toolbox."""
        return 'Raster Calculator'

    def longName(self):
        """Detailed provider description."""
        return 'Raster Calculator - Map Algebra Expression Tool'

    def icon(self):
        """Provider icon shown in Processing Toolbox."""
        icon_path = os.path.join(os.path.dirname(__file__), 'toolbox.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)

    def load(self):
        """Called when provider is first loaded."""
        self.refreshAlgorithms()
        return True

    def unload(self):
        """Called when provider is unloaded."""
        pass

    def supportedOutputRasterLayerExtensions(self):
        """Define supported output raster formats."""
        return ['tif', 'tiff']
