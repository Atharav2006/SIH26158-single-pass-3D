"""
Tests for Depth/AI Quality Analyzer (Member 3 TASK 8)
"""

import math
import numpy as np
import pytest
from src.quality.depth_ai_analyzer import (
    DepthAIAnalyzer,
    DepthEdgeStatistics,
    DepthOutlierStatistics,
    AIQualityStatistics,
    generate_depth_ai_quality_report
)

def test_depth_edge_empty():
    analyzer = DepthAIAnalyzer()
    stats = analyzer.analyze_depth_edges([])
    assert stats.availability == "NOT_AVAILABLE"

def test_depth_edge_valid():
    analyzer = DepthAIAnalyzer()
    # Create simple depth bump
    dmap = np.ones((5, 5))
    dmap[:, 2] = 2.0  # create edge
    
    stats = analyzer.analyze_depth_edges([dmap], high_frequency_threshold=0.1)
    assert stats.availability == "AVAILABLE"
    assert stats.edge_energy_mean > 0.0
    assert stats.discontinuity_ratio > 0.0
    
def test_depth_outlier_empty():
    analyzer = DepthAIAnalyzer()
    stats = analyzer.analyze_depth_outliers([np.array([])])
    assert stats.availability == "AVAILABLE"
    assert stats.outlier_count == 0

def test_depth_outlier_valid():
    analyzer = DepthAIAnalyzer()
    # Base array around median 1.0
    # Add a massive outlier to trigger MAD isolation
    dmap = np.array([1.0, 1.1, 0.9, 1.0, 1.1, 100.0, 1.0])
    stats = analyzer.analyze_depth_outliers([dmap], mad_threshold=3.0)
    
    assert stats.availability == "AVAILABLE"
    assert stats.outlier_count == 1
    # 1 out of 7 is ~0.142
    assert math.isclose(stats.outlier_ratio, 1.0/7.0, abs_tol=1e-4)

def test_depth_outlier_nan_negative():
    analyzer = DepthAIAnalyzer()
    
    # Should isolate valid values and ignore nan/-1
    dmap = np.array([1.0, 1.1, 0.9, -1.0, np.nan, np.inf, 10.0])
    stats = analyzer.analyze_depth_outliers([dmap], mad_threshold=3.0)
    
    assert stats.outlier_count == 1  # only 10.0 is the valid outlier

def test_depth_ai_build_report():
    analyzer = DepthAIAnalyzer()
    
    e = DepthEdgeStatistics(edge_energy_mean=1.5, discontinuity_ratio=0.8, availability="AVAILABLE")
    o = DepthOutlierStatistics(outlier_count=50, outlier_ratio=0.2, mad=0.5, availability="AVAILABLE")
    a = AIQualityStatistics()
    
    report = analyzer.build_report(e, o, a)
    
    assert report.quality == "POOR"
    assert any("Highly discontinuous" in w for w in report.warnings)
    assert any("High localized depth outliers" in w for w in report.warnings)
    
def test_depth_ai_human_report():
    analyzer = DepthAIAnalyzer()
    e = DepthEdgeStatistics(availability="AVAILABLE")
    o = DepthOutlierStatistics(availability="NOT_AVAILABLE")
    a = AIQualityStatistics()
    
    report = analyzer.build_report(e, o, a)
    output = generate_depth_ai_quality_report(report)
    
    assert "DEPTH / AI RECONSTRUCTION QUALITY REPORT" in output
    assert "Multi-View Consistency    : NOT_AVAILABLE" in output
