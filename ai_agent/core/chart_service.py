"""Chart and visualization generation service.

This module provides:
- Multiple chart types (bar, line, pie, scatter, area)
- Output formats (PNG, SVG, HTML interactive, base64)
- Data-driven chart generation from query results
- Chart styling and theming
"""
import json
import logging
import base64
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from datetime import datetime

log = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    DONUT = "donut"
    HORIZONTAL_BAR = "horizontal_bar"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"


class OutputFormat(str, Enum):
    """Chart output formats."""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"  # Vega-Lite spec
    BASE64 = "base64"


@dataclass
class ChartConfig:
    """Chart configuration."""
    chart_type: ChartType
    title: str
    x_column: Optional[str] = None  # Category/Date column
    y_column: Optional[str] = None  # Value column
    color_column: Optional[str] = None  # For coloring/grouping
    
    # Styling
    width: int = 800
    height: int = 400
    theme: str = "light"  # light/dark
    primary_color: Optional[str] = None
    
    # Axes
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    show_grid: bool = True
    show_legend: bool = True
    
    # Formatting
    value_format: Optional[str] = None  # e.g., "currency", "percent", "number"
    currency_symbol: str = "$"
    decimals: int = 2
    
    # Data
    data: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = []


class ChartService:
    """Service for generating charts and visualizations."""
    
    # Color palettes
    PALETTES = {
        "default": ["#4F46E5", "#06B6D4", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981"],
        "blue": ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
        "green": ["#065F46", "#059669", "#10B981", "#34D399", "#6EE7B7"],
        "warm": ["#DC2626", "#EA580C", "#D97706", "#CA8A04", "#65A30D"],
    }
    
    def __init__(self):
        """Initialize chart service."""
        self._matplotlib_available = False
        self._plotly_available = False
        self._vega_available = False
        
        self._try_imports()
    
    def _try_imports(self):
        """Try to import optional visualization libraries."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self._plt = plt
            self._matplotlib_available = True
            log.info("ChartService: matplotlib available")
        except ImportError:
            log.info("ChartService: matplotlib not available")
        
        try:
            import plotly
            import plotly.graph_objects as go
            self._go = go
            self._plotly_available = True
            log.info("ChartService: plotly available")
        except ImportError:
            log.info("ChartService: plotly not available")
        
        try:
            import altair as alt
            self._alt = alt
            self._vega_available = True
            log.info("ChartService: altair/vega available")
        except ImportError:
            log.info("ChartService: altair not available")
    
    def generate_chart(
        self,
        config: ChartConfig,
        output_format: OutputFormat = OutputFormat.PNG,
    ) -> Union[bytes, str, Dict]:
        """Generate a chart from configuration.
        
        Args:
            config: Chart configuration
            output_format: Desired output format
            
        Returns:
            Chart data in requested format:
            - PNG: bytes
            - SVG: str
            - HTML: str
            - JSON: dict (Vega-Lite spec)
            - BASE64: str (base64 encoded PNG)
        """
        if not config.data:
            raise ValueError("Chart data is required")
        
        # Use plotly for interactive charts, matplotlib for static
        if output_format == OutputFormat.HTML:
            return self._generate_plotly_chart(config, output_format)
        elif output_format == OutputFormat.JSON:
            return self._generate_vega_chart(config)
        else:
            return self._generate_matplotlib_chart(config, output_format)
    
    def _generate_matplotlib_chart(
        self,
        config: ChartConfig,
        output_format: OutputFormat,
    ) -> Union[bytes, str]:
        """Generate chart using matplotlib."""
        if not self._matplotlib_available:
            raise RuntimeError("matplotlib not available. Install: pip install matplotlib")
        
        plt = self._plt
        
        # Prepare data
        df = self._data_to_dataframe(config.data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width / 100, config.height / 100))
        
        # Set theme
        if config.theme == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # Get colors
        colors = self.PALETTES.get("default")
        if config.primary_color:
            colors = [config.primary_color] + colors
        
        # Generate chart based on type
        if config.chart_type == ChartType.BAR:
            self._create_bar_chart(ax, df, config, colors)
        elif config.chart_type == ChartType.HORIZONTAL_BAR:
            self._create_horizontal_bar_chart(ax, df, config, colors)
        elif config.chart_type == ChartType.LINE:
            self._create_line_chart(ax, df, config, colors)
        elif config.chart_type == ChartType.PIE:
            self._create_pie_chart(ax, df, config, colors)
        elif config.chart_type == ChartType.AREA:
            self._create_area_chart(ax, df, config, colors)
        elif config.chart_type == ChartType.SCATTER:
            self._create_scatter_chart(ax, df, config, colors)
        else:
            # Default to bar
            self._create_bar_chart(ax, df, config, colors)
        
        # Styling
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        if config.show_legend and ax.get_legend():
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Output
        buffer = BytesIO()
        
        if output_format == OutputFormat.SVG:
            plt.savefig(buffer, format='svg', bbox_inches='tight')
            plt.close(fig)
            return buffer.getvalue().decode('utf-8')
        elif output_format == OutputFormat.BASE64:
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:  # PNG
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            return buffer.getvalue()
    
    def _generate_plotly_chart(
        self,
        config: ChartConfig,
        output_format: OutputFormat,
    ) -> str:
        """Generate interactive chart using plotly."""
        if not self._plotly_available:
            raise RuntimeError("plotly not available. Install: pip install plotly")
        
        go = self._go
        df = self._data_to_dataframe(config.data)
        
        fig = go.Figure()
        
        if config.chart_type == ChartType.BAR:
            fig.add_trace(go.Bar(
                x=df[config.x_column] if config.x_column else df.index,
                y=df[config.y_column] if config.y_column else df.iloc[:, 0],
                marker_color=self.PALETTES["default"],
            ))
        elif config.chart_type == ChartType.LINE:
            fig.add_trace(go.Line(
                x=df[config.x_column] if config.x_column else df.index,
                y=df[config.y_column] if config.y_column else df.iloc[:, 0],
            ))
        elif config.chart_type == ChartType.PIE:
            fig.add_trace(go.Pie(
                labels=df[config.x_column] if config.x_column else df.index,
                values=df[config.y_column] if config.y_column else df.iloc[:, 0],
            ))
        elif config.chart_type == ChartType.SCATTER:
            fig.add_trace(go.Scatter(
                x=df[config.x_column] if config.x_column else df.index,
                y=df[config.y_column] if config.y_column else df.iloc[:, 0],
                mode='markers',
            ))
        
        # Layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template='plotly_white' if config.theme == 'light' else 'plotly_dark',
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def _generate_vega_chart(self, config: ChartConfig) -> Dict:
        """Generate Vega-Lite spec."""
        if not self._vega_available:
            # Return manual spec without altair
            return self._create_manual_vega_spec(config)
        
        import altair as alt
        
        df = self._data_to_dataframe(config.data)
        
        chart = alt.Chart(df)
        
        if config.chart_type == ChartType.BAR:
            chart = chart.mark_bar().encode(
                x=alt.X(config.x_column or df.columns[0], title=config.x_label),
                y=alt.Y(config.y_column or df.columns[1], title=config.y_label),
            )
        elif config.chart_type == ChartType.LINE:
            chart = chart.mark_line().encode(
                x=alt.X(config.x_column or df.columns[0]),
                y=alt.Y(config.y_column or df.columns[1]),
            )
        elif config.chart_type == ChartType.SCATTER:
            chart = chart.mark_point().encode(
                x=alt.X(config.x_column or df.columns[0]),
                y=alt.Y(config.y_column or df.columns[1]),
            )
        
        chart = chart.properties(
            title=config.title,
            width=config.width,
            height=config.height,
        )
        
        return chart.to_dict()
    
    def _create_manual_vega_spec(self, config: ChartConfig) -> Dict:
        """Create Vega-Lite spec without altair."""
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "data": {"values": config.data},
        }
        
        x_field = config.x_column or list(config.data[0].keys())[0]
        y_field = config.y_column or list(config.data[0].keys())[1]
        
        if config.chart_type == ChartType.BAR:
            spec["mark"] = "bar"
            spec["encoding"] = {
                "x": {"field": x_field, "type": "ordinal", "title": config.x_label or x_field},
                "y": {"field": y_field, "type": "quantitative", "title": config.y_label or y_field},
            }
        elif config.chart_type == ChartType.LINE:
            spec["mark"] = {"type": "line", "point": True}
            spec["encoding"] = {
                "x": {"field": x_field, "type": "ordinal"},
                "y": {"field": y_field, "type": "quantitative"},
            }
        elif config.chart_type == ChartType.PIE:
            spec["mark"] = {"type": "arc", "innerRadius": 50 if config.chart_type == ChartType.DONUT else 0}
            spec["encoding"] = {
                "theta": {"field": y_field, "type": "quantitative"},
                "color": {"field": x_field, "type": "nominal"},
            }
        elif config.chart_type == ChartType.SCATTER:
            spec["mark"] = "point"
            spec["encoding"] = {
                "x": {"field": x_field, "type": "quantitative"},
                "y": {"field": y_field, "type": "quantitative"},
            }
        
        return spec
    
    def _data_to_dataframe(self, data: List[Dict]) -> Any:
        """Convert data to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            # Return as-is if pandas not available
            return data
    
    def _create_bar_chart(self, ax, df, config, colors):
        """Create vertical bar chart."""
        x = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        y = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.bar(x, y, color=colors[0])
    
    def _create_horizontal_bar_chart(self, ax, df, config, colors):
        """Create horizontal bar chart."""
        x = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        y = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.barh(x, y, color=colors[0])
    
    def _create_line_chart(self, ax, df, config, colors):
        """Create line chart."""
        x = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        y = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.plot(x, y, color=colors[0], marker='o')
    
    def _create_pie_chart(self, ax, df, config, colors):
        """Create pie chart."""
        labels = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        sizes = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%')
    
    def _create_area_chart(self, ax, df, config, colors):
        """Create area chart."""
        x = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        y = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.fill_between(x, y, alpha=0.5, color=colors[0])
        ax.plot(x, y, color=colors[0])
    
    def _create_scatter_chart(self, ax, df, config, colors):
        """Create scatter chart."""
        x = [d.get(config.x_column) or d.get(list(d.keys())[0]) for d in config.data]
        y = [d.get(config.y_column) or d.get(list(d.keys())[1]) for d in config.data]
        ax.scatter(x, y, color=colors[0], alpha=0.6)
    
    def infer_chart_config(
        self,
        data: List[Dict[str, Any]],
        title: str,
        user_query: Optional[str] = None,
    ) -> ChartConfig:
        """Infer chart configuration from data and optional query.
        
        This is useful for automatic chart generation from query results.
        """
        if not data:
            raise ValueError("Cannot infer chart config from empty data")
        
        columns = list(data[0].keys())
        
        # Detect column types
        numeric_cols = []
        categorical_cols = []
        date_cols = []
        
        for col in columns:
            sample = data[0].get(col)
            if isinstance(sample, (int, float)):
                numeric_cols.append(col)
            elif isinstance(sample, str):
                # Check if it looks like a date
                if any(keyword in col.lower() for keyword in ['date', 'time', 'at', 'on']):
                    date_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Infer chart type from query keywords
        chart_type = ChartType.BAR
        if user_query:
            query_lower = user_query.lower()
            if any(w in query_lower for w in ['trend', 'over time', 'growth', 'history']):
                chart_type = ChartType.LINE if date_cols else ChartType.BAR
            elif any(w in query_lower for w in ['distribution', 'share', 'percentage', 'breakdown']):
                chart_type = ChartType.PIE if len(data) <= 8 else ChartType.BAR
            elif any(w in query_lower for w in ['compare', 'versus', 'vs']):
                chart_type = ChartType.GROUPED_BAR if len(numeric_cols) > 1 else ChartType.BAR
            elif any(w in query_lower for w in ['correlation', 'relationship']):
                chart_type = ChartType.SCATTER if len(numeric_cols) >= 2 else ChartType.BAR
        
        # Select columns
        x_col = None
        y_col = None
        
        if categorical_cols:
            x_col = categorical_cols[0]
        elif date_cols:
            x_col = date_cols[0]
        
        if numeric_cols:
            y_col = numeric_cols[0]
        
        return ChartConfig(
            chart_type=chart_type,
            title=title,
            x_column=x_col,
            y_column=y_col,
            data=data,
        )


# Global instance
_chart_service: Optional[ChartService] = None


def get_chart_service() -> ChartService:
    """Get or create the chart service singleton."""
    global _chart_service
    if _chart_service is None:
        _chart_service = ChartService()
    return _chart_service


def generate_chart_from_data(
    data: List[Dict[str, Any]],
    title: str,
    chart_type: ChartType = ChartType.BAR,
    output_format: OutputFormat = OutputFormat.PNG,
    **kwargs
) -> Union[bytes, str, Dict]:
    """Convenience function to generate a chart from data."""
    service = get_chart_service()
    config = ChartConfig(
        chart_type=chart_type,
        title=title,
        data=data,
        **kwargs
    )
    return service.generate_chart(config, output_format)


def generate_chart_from_query_result(
    data: List[Dict[str, Any]],
    title: str,
    user_query: Optional[str] = None,
    output_format: OutputFormat = OutputFormat.PNG,
) -> Union[bytes, str, Dict]:
    """Auto-generate chart from SQL query result."""
    service = get_chart_service()
    config = service.infer_chart_config(data, title, user_query)
    return service.generate_chart(config, output_format)
