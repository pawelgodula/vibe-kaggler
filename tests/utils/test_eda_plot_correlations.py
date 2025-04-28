# Tests for eda_plot_correlations functions.

import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import base64
import numpy as np

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_plot_correlations import plot_correlation_heatmap, plot_target_correlation_bar
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_plot_correlations import plot_correlation_heatmap, plot_target_correlation_bar

# Mock plotting functions to check calls and avoid actual plot generation
@patch('utils.eda_plot_correlations.plt.figure')
@patch('utils.eda_plot_correlations.sns.heatmap')
@patch('utils.eda_plot_correlations.plt.savefig')
@patch('utils.eda_plot_correlations.plt.close')
class TestPlotCorrelationHeatmap(unittest.TestCase):

    def setUp(self):
        # Sample correlation matrix (output from calculate_correlations)
        self.feature_corr_matrix = pl.DataFrame({
            'feature': ['f1', 'f2', 'f3'],
            'f1': [1.0, 0.5, -0.2],
            'f2': [0.5, 1.0, 0.1],
            'f3': [-0.2, 0.1, 1.0]
        })

    def test_plot_heatmap_runs(self, mock_close, mock_savefig, mock_heatmap, mock_figure):
        """Test if heatmap plotting runs and returns base64."""
        result = plot_correlation_heatmap(self.feature_corr_matrix)
        
        self.assertTrue(isinstance(result, str))
        try:
            base64.b64decode(result)
            is_base64 = True
        except Exception:
            is_base64 = False
        self.assertTrue(is_base64, "Result should be a valid base64 string")
        
        # Relax check: Ensure it was called at least once
        self.assertTrue(mock_figure.call_count >= 1)
        mock_heatmap.assert_called_once()
        # Check if feature names are passed correctly (last arg to heatmap usually)
        call_args, call_kwargs = mock_heatmap.call_args
        self.assertEqual(call_kwargs.get('xticklabels'), ['f1', 'f2', 'f3'])
        self.assertEqual(call_kwargs.get('yticklabels'), ['f1', 'f2', 'f3'])
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_plot_heatmap_annot_small(self, mock_close, mock_savefig, mock_heatmap, mock_figure):
        """Test annotation is enabled for small matrices by default."""
        plot_correlation_heatmap(self.feature_corr_matrix, annot=True)
        call_args, call_kwargs = mock_heatmap.call_args
        self.assertTrue(call_kwargs.get('annot'))

    def test_plot_heatmap_annot_large(self, mock_close, mock_savefig, mock_heatmap, mock_figure):
        """Test annotation is disabled for large matrices even if requested."""
        large_matrix = pl.DataFrame({
            'feature': [f'f{i}' for i in range(25)],
            **{f'f{i}': [0.0]*25 for i in range(25)} # Dummy data
        })
        plot_correlation_heatmap(large_matrix, annot=True, max_size_for_annot=20)
        call_args, call_kwargs = mock_heatmap.call_args
        self.assertFalse(call_kwargs.get('annot'))

    def test_plot_heatmap_none_input(self, mock_close, mock_savefig, mock_heatmap, mock_figure):
        """Test heatmap plotting with None input."""
        result = plot_correlation_heatmap(None)
        self.assertEqual(result, "")
        mock_figure.assert_not_called()

    def test_plot_heatmap_empty_input(self, mock_close, mock_savefig, mock_heatmap, mock_figure):
        """Test heatmap plotting with empty DataFrame input."""
        result = plot_correlation_heatmap(pl.DataFrame())
        self.assertEqual(result, "")
        mock_figure.assert_not_called()

@patch('utils.eda_plot_correlations.plt.figure')
@patch('utils.eda_plot_correlations.sns.barplot')
@patch('utils.eda_plot_correlations.plt.savefig')
@patch('utils.eda_plot_correlations.plt.close')
class TestPlotTargetCorrelationBar(unittest.TestCase):

    def setUp(self):
        # Sample target correlations (output from calculate_correlations)
        self.target_corr = pl.DataFrame({
            'feature': ['f1', 'f3', 'f2'],
            'target_correlation': [0.8, 0.5, -0.3]
        }).sort('target_correlation', descending=True)

    def test_plot_bar_runs(self, mock_close, mock_savefig, mock_barplot, mock_figure):
        """Test if bar plotting runs and returns base64."""
        result = plot_target_correlation_bar(self.target_corr, target_col='target')
        
        self.assertTrue(isinstance(result, str))
        try:
            base64.b64decode(result)
            is_base64 = True
        except Exception:
            is_base64 = False
        self.assertTrue(is_base64, "Result should be a valid base64 string")
        
        # Relax check: Ensure it was called at least once
        self.assertTrue(mock_figure.call_count >= 1)
        mock_barplot.assert_called_once()
        call_args, call_kwargs = mock_barplot.call_args
        # Check if data passed is sorted correctly (descending by correlation)
        self.assertEqual(call_kwargs['data']['feature'].tolist(), ['f1', 'f3', 'f2'])
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_plot_bar_max_features(self, mock_close, mock_savefig, mock_barplot, mock_figure):
        """Test limiting features in the bar plot."""
        target_corr_large = pl.DataFrame({
            'feature': [f'f{i}' for i in range(10)],
            'target_correlation': np.linspace(0.9, -0.9, 10)
        })
        plot_target_correlation_bar(target_corr_large, target_col='target', max_features_to_plot=5)
        call_args, call_kwargs = mock_barplot.call_args
        self.assertEqual(len(call_kwargs['data']), 5) 
        # Check if top 5 by absolute value are selected, then sorted by original value descending.
        # AbsCorrs: f0(0.9), f9(0.9), f1(0.7), f8(0.7), f2(0.5), f7(0.5), ...
        # Top 5 (abs): f0, f9, f1, f8, f7 (assuming tie-break favouring lower index for positive, higher for negative)
        # Sorted (orig value desc): f0(0.9), f1(0.7), f7(-0.5), f8(-0.7), f9(-0.9)
        self.assertEqual(call_kwargs['data']['feature'].tolist(), ['f0', 'f1', 'f7', 'f8', 'f9'])

    def test_plot_bar_none_input(self, mock_close, mock_savefig, mock_barplot, mock_figure):
        """Test bar plotting with None input."""
        result = plot_target_correlation_bar(None, target_col='target')
        self.assertEqual(result, "")
        mock_figure.assert_not_called()

    def test_plot_bar_empty_input(self, mock_close, mock_savefig, mock_barplot, mock_figure):
        """Test bar plotting with empty DataFrame input."""
        result = plot_target_correlation_bar(pl.DataFrame(), target_col='target')
        self.assertEqual(result, "")
        mock_figure.assert_not_called()

    def test_plot_bar_missing_cols(self, mock_close, mock_savefig, mock_barplot, mock_figure):
        """Test bar plotting with missing expected columns."""
        bad_df = pl.DataFrame({'features': ['a'], 'corr': [0.5]})
        result = plot_target_correlation_bar(bad_df, target_col='target')
        self.assertEqual(result, "")
        mock_figure.assert_not_called()


if __name__ == '__main__':
    unittest.main() 