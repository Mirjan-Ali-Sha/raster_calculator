# -*- coding: utf-8 -*-
"""
Raster Calculator Processing Algorithm
"""

import os
import re
import numpy as np
from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterString,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterRasterDestination,
    QgsProcessing,
    QgsProcessingException,
    QgsRasterLayer
)


# Define all custom functions for raster calculations
def _Con(condition, true_val, false_val):
    """Conditional: returns true_val where condition is True, else false_val."""
    return np.where(condition, true_val, false_val)

def _SetNull(condition, false_val):
    """SetNull: returns NaN where condition is True, else false_val."""
    return np.where(condition, np.nan, false_val)

def _Pick(value, *options):
    """Pick: selects from options based on value (1-indexed)."""
    result = np.zeros_like(value, dtype=np.float64)
    for i, opt in enumerate(options, 1):
        mask = (value == i)
        if isinstance(opt, np.ndarray):
            result[mask] = opt[mask]
        else:
            result[mask] = opt
    return result

def _Exp10(x):
    """10^x function."""
    return np.power(10.0, x)

def _Exp2(x):
    """2^x function."""
    return np.power(2.0, x)

def _Float(x):
    """Convert to float."""
    return x.astype(np.float64) if isinstance(x, np.ndarray) else float(x)

def _Int(x):
    """Convert to integer (truncate)."""
    return np.trunc(x).astype(np.int32) if isinstance(x, np.ndarray) else int(x)

def _Ln(x):
    """Natural logarithm."""
    return np.log(x)

def _Log2(x):
    """Base-2 logarithm."""
    return np.log2(x)

def _Mod(x, y):
    """Modulo operation."""
    return np.mod(x, y)

def _Power(base, exp):
    """Power function: base^exp."""
    return np.power(base, exp)

def _Square(x):
    """Square of x."""
    return np.square(x)

def _SquareRoot(x):
    """Square root of x."""
    return np.sqrt(x)

def _RoundDown(x):
    """Round down (floor)."""
    return np.floor(x)

def _RoundUp(x):
    """Round up (ceiling)."""
    return np.ceil(x)

def _Round(x):
    """Round to nearest integer."""
    return np.round(x)

def _Diff(x, y):
    """Absolute difference: |x - y|."""
    return np.abs(x - y)

def _InList(value, *options):
    """InList: returns 1 if value is in the list, else 0."""
    result = np.zeros_like(value, dtype=np.float64)
    for opt in options:
        result = np.where(value == opt, 1.0, result)
    return result

def _IsNull(x):
    """IsNull: returns 1 where x is NaN/NoData, else 0."""
    return np.where(np.isnan(x), 1.0, 0.0)

def _Over(x, y):
    """Over: returns x where x is not NaN, else y."""
    return np.where(~np.isnan(x), x, y)

def _Test(condition, true_val):
    """Test: returns true_val where condition is True, else 0."""
    return np.where(condition, true_val, 0.0)

def _ATan2(y, x):
    """ATan2: two-argument arctangent."""
    return np.arctan2(y, x)


# Function replacements mapping
FUNCTION_REPLACEMENTS = {
    # Conditional
    'Con': '_Con',
    'SetNull': '_SetNull',
    'Pick': '_Pick',
    # Math
    'Abs': 'np.abs',
    'Exp': 'np.exp',
    'Exp10': '_Exp10',
    'Exp2': '_Exp2',
    'Float': '_Float',
    'Int': '_Int',
    'Ln': '_Ln',
    'Log': 'np.log',
    'Log10': 'np.log10',
    'Log2': '_Log2',
    'Mod': '_Mod',
    'Power': '_Power',
    'Square': '_Square',
    'SquareRoot': '_SquareRoot',
    'Sqrt': 'np.sqrt',
    'Round': '_Round',
    'RoundDown': '_RoundDown',
    'RoundUp': '_RoundUp',
    'Floor': 'np.floor',
    'Ceil': 'np.ceil',
    # Trigonometric
    'Sin': 'np.sin',
    'Cos': 'np.cos',
    'Tan': 'np.tan',
    'ASin': 'np.arcsin',
    'ACos': 'np.arccos',
    'ATan': 'np.arctan',
    'ATan2': '_ATan2',
    'SinH': 'np.sinh',
    'CosH': 'np.cosh',
    'TanH': 'np.tanh',
    'ASinH': 'np.arcsinh',
    'ACosH': 'np.arccosh',
    'ATanH': 'np.arctanh',
    # Logical/Other
    'Diff': '_Diff',
    'InList': '_InList',
    'IsNull': '_IsNull',
    'Over': '_Over',
    'Test': '_Test',
}

# Evaluation context
EVAL_CONTEXT = {
    'np': np,
    '_Con': _Con,
    '_SetNull': _SetNull,
    '_Pick': _Pick,
    '_Exp10': _Exp10,
    '_Exp2': _Exp2,
    '_Float': _Float,
    '_Int': _Int,
    '_Ln': _Ln,
    '_Log2': _Log2,
    '_Mod': _Mod,
    '_Power': _Power,
    '_Square': _Square,
    '_SquareRoot': _SquareRoot,
    '_RoundDown': _RoundDown,
    '_RoundUp': _RoundUp,
    '_Round': _Round,
    '_Diff': _Diff,
    '_InList': _InList,
    '_IsNull': _IsNull,
    '_Over': _Over,
    '_Test': _Test,
    '_ATan2': _ATan2,
}


class RasterCalculatorAlgorithm(QgsProcessingAlgorithm):
    """Raster Calculator Algorithm for map algebra expressions."""
    
    EXPRESSION = 'EXPRESSION'
    INPUT_LAYERS = 'INPUT_LAYERS'
    OUTPUT = 'OUTPUT'
    
    def __init__(self):
        super().__init__()
    
    def tr(self, string):
        return QCoreApplication.translate('RasterCalculatorAlgorithm', string)
    
    def createInstance(self):
        return RasterCalculatorAlgorithm()
    
    def name(self):
        return 'rastercalculator'
    
    def displayName(self):
        return self.tr('Raster Calculator')
    
    def group(self):
        return self.tr('MAS Raster Processing')
    
    def groupId(self):
        return 'mas_raster_processing'
    
    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'raster_calculator.png')
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()
    
    def shortHelpString(self):
        return self.tr("""
Performs map algebra calculations on raster layers.

<b>Expression Format:</b>
"layername@1" for band 1, "layername@2" for band 2

<b>Operators:</b> + - * / ^ == != > >= < <= & | ~

<b>Conditional:</b> Con(cond, true, false), SetNull(cond, val), Pick(val, opt1, opt2...)

<b>Math:</b> Abs, Exp, Exp10, Exp2, Float, Int, Ln, Log10, Log2, Mod(x,y), Power(b,e), Square, SquareRoot, Round, RoundDown, RoundUp

<b>Trig:</b> Sin, Cos, Tan, ASin, ACos, ATan, ATan2(y,x), SinH, CosH, TanH, ASinH, ACosH, ATanH

<b>Other:</b> Diff(x,y), InList(val, v1, v2...), IsNull(x), Over(x,y), Test(cond, val)

<b>Examples:</b>
- "dem@1" + 100
- Con("dem@1" > 500, 1, 0)
- Power("dem@1", 2)
        """)
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterString(
                self.EXPRESSION,
                self.tr('Map Algebra Expression'),
                defaultValue='',
                multiLine=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_LAYERS,
                self.tr('Input Raster Layers'),
                QgsProcessing.TypeRaster,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Output Raster')
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        expression = self.parameterAsString(parameters, self.EXPRESSION, context)
        input_layers = self.parameterAsLayerList(parameters, self.INPUT_LAYERS, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        
        if not expression:
            raise QgsProcessingException("Expression cannot be empty")
        
        # Build layer references
        layer_refs = {}
        for layer in input_layers:
            if isinstance(layer, QgsRasterLayer):
                layer_refs[layer.name()] = layer.source()
        
        # Find layers from project - exclude @ from layer name
        layer_pattern = r'"([^"@]+)(?:@(\d+))?"'
        matches = re.findall(layer_pattern, expression)
        
        for layer_name, _ in matches:
            if layer_name not in layer_refs:
                from qgis.core import QgsProject
                layer = QgsProject.instance().mapLayersByName(layer_name)
                if layer and isinstance(layer[0], QgsRasterLayer):
                    layer_refs[layer_name] = layer[0].source()
        
        feedback.pushInfo(f"Expression: {expression}")
        feedback.pushInfo(f"Found {len(layer_refs)} layer references")
        
        self.calculate_raster(expression, layer_refs, output_path, feedback)
        
        return {self.OUTPUT: output_path}
    
    def calculate_raster(self, expression, layer_refs, output_path, feedback):
        """Perform the raster calculation."""
        feedback.setProgress(10)
        
        # Pattern: "layername" or "layername@band" - exclude @ from layer name
        layer_pattern = r'"([^"@]+)(?:@(\d+))?"'
        matches = re.findall(layer_pattern, expression)
        
        if not matches:
            raise QgsProcessingException(
                "No raster layers found. Use \"layername@1\" format."
            )
        
        datasets = {}
        arrays = {}
        reference_ds = None
        
        feedback.setProgress(20)
        
        for layer_name, band_str in matches:
            if feedback.isCanceled():
                return
            
            if layer_name not in layer_refs:
                raise QgsProcessingException(f"Layer '{layer_name}' not found")
            
            file_path = layer_refs[layer_name]
            band = int(band_str) if band_str else 1
            key = f'"{layer_name}@{band}"' if band_str else f'"{layer_name}"'
            
            if file_path not in datasets:
                ds = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds is None:
                    raise QgsProcessingException(f"Cannot open: {file_path}")
                datasets[file_path] = ds
                if reference_ds is None:
                    reference_ds = ds
            
            ds = datasets[file_path]
            if band > ds.RasterCount:
                raise QgsProcessingException(f"Band {band} not in '{layer_name}'")
            
            rb = ds.GetRasterBand(band)
            arr = rb.ReadAsArray().astype(np.float64)
            
            nodata = rb.GetNoDataValue()
            if nodata is not None:
                arr[arr == nodata] = np.nan
            
            arrays[key] = arr
            simple_key = f'"{layer_name}"'
            if simple_key not in arrays:
                arrays[simple_key] = arr
        
        if reference_ds is None:
            raise QgsProcessingException("No valid rasters found")
        
        feedback.setProgress(50)
        
        # Prepare expression
        eval_expr = expression
        for key in arrays.keys():
            eval_expr = re.sub(re.escape(key), f'_arrays[{repr(key)}]', eval_expr)
        
        eval_expr = eval_expr.replace('^', '**')
        eval_expr = eval_expr.replace('&', ' & ')
        eval_expr = eval_expr.replace('|', ' | ')
        eval_expr = eval_expr.replace('~', ' ~')
        
        for func, replacement in FUNCTION_REPLACEMENTS.items():
            pattern = r'\b' + func + r'\s*\('
            eval_expr = re.sub(pattern, replacement + '(', eval_expr, flags=re.IGNORECASE)
        
        context = EVAL_CONTEXT.copy()
        context['_arrays'] = arrays
        
        try:
            result = eval(eval_expr, {"__builtins__": {}}, context)
        except Exception as e:
            raise QgsProcessingException(f"Evaluation error: {str(e)}")
        
        if feedback.isCanceled():
            return
        
        feedback.setProgress(80)
        
        cols = reference_ds.RasterXSize
        rows = reference_ds.RasterYSize
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path, cols, rows, 1, gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        if out_ds is None:
            raise QgsProcessingException(f"Cannot create: {output_path}")
        
        out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
        out_ds.SetProjection(reference_ds.GetProjection())
        
        out_band = out_ds.GetRasterBand(1)
        nodata_out = -9999.0
        
        if isinstance(result, np.ndarray):
            result = result.astype(np.float32)
            result[np.isnan(result)] = nodata_out
        
        out_band.SetNoDataValue(nodata_out)
        out_band.WriteArray(result)
        out_ds.BuildOverviews('NEAREST', [2, 4, 8, 16])
        out_band.FlushCache()
        out_ds = None
        
        for ds in datasets.values():
            ds = None
        
        feedback.setProgress(100)
        feedback.pushInfo(f"Output: {output_path}")
