# Tests for eda_plot_feature_summary function.

import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import base64

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_plot_feature_summary import plot_feature_summary
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_plot_feature_summary import plot_feature_summary

@patch('utils.eda_plot_feature_summary.plt.subplots') # Mock subplots to get mock axes
@patch('utils.eda_plot_feature_summary.sns.lineplot')
@patch('utils.eda_plot_feature_summary.sns.barplot') # Used for count on secondary axis
@patch('utils.eda_plot_feature_summary.plt.savefig')
@patch('utils.eda_plot_feature_summary.plt.close')
class TestPlotFeatureSummary(unittest.TestCase):

    def setUp(self):
        # Sample summary df (output from analyze_target_by_feature)
        self.summary_df = pl.DataFrame({
            'feature_value': ['A', 'B', 'C', 'D'],
            'count': [100, 50, 120, 30],
            'target_mean': [10.5, 12.1, 9.8, 15.0],
            'target_median': [10.0, 11.8, 9.5, 14.8]
        })

    def test_plot_summary_runs_mean(self, mock_close, mock_savefig, mock_barplot, mock_lineplot, mock_subplots):
        """Test plotting mean summary runs and returns base64."""
        # Mock subplots to return a figure and two axes (for twinx)
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax1.twinx.return_value = mock_ax2
        mock_subplots.return_value = (mock_fig, mock_ax1)

        result = plot_feature_summary(self.summary_df, 'cat_feat', 'target', plot_metric='mean', include_count=True)
        
        self.assertTrue(isinstance(result, str))
        try:
            base64.b64decode(result)
            is_base64 = True
        except Exception:
            is_base64 = False
        self.assertTrue(is_base64, "Result should be a valid base64 string")
        
        mock_subplots.assert_called_once()
        mock_lineplot.assert_called_once()
        mock_barplot.assert_called_once() # Count plot
        mock_ax1.twinx.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_any_call(mock_fig)

        # Check if correct metric column is used for lineplot
        call_args, call_kwargs = mock_lineplot.call_args
        self.assertEqual(call_kwargs.get('y'), 'target_mean')

    def test_plot_summary_runs_median_no_count(self, mock_close, mock_savefig, mock_barplot, mock_lineplot, mock_subplots):
        """Test plotting median summary without count runs and returns base64."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax1)

        result = plot_feature_summary(self.summary_df, 'cat_feat', 'target', plot_metric='median', include_count=False)
        
        self.assertTrue(isinstance(result, str))
        try:
            base64.b64decode(result)
            is_base64 = True
        except Exception:
            is_base64 = False
        self.assertTrue(is_base64, "Result should be a valid base64 string")
        
        mock_subplots.assert_called_once()
        mock_lineplot.assert_called_once()
        mock_barplot.assert_not_called() # Count plot should not be called
        mock_ax1.twinx.assert_not_called()
        mock_savefig.assert_called_once()
        mock_close.assert_any_call(mock_fig)

        # Check if correct metric column is used for lineplot
        call_args, call_kwargs = mock_lineplot.call_args
        self.assertEqual(call_kwargs.get('y'), 'target_median')

    def test_plot_summary_none_input(self, mock_close, mock_savefig, mock_barplot, mock_lineplot, mock_subplots):
        """Test plotting with None input."""
        result = plot_feature_summary(None, 'feat', 'target')
        self.assertEqual(result, "")
        mock_subplots.assert_not_called()

    def test_plot_summary_empty_input(self, mock_close, mock_savefig, mock_barplot, mock_lineplot, mock_subplots):
        """Test plotting with empty DataFrame input."""
        result = plot_feature_summary(pl.DataFrame(), 'feat', 'target')
        self.assertEqual(result, "")
        mock_subplots.assert_not_called()

    def test_plot_summary_missing_cols(self, mock_close, mock_savefig, mock_barplot, mock_lineplot, mock_subplots):
        """Test plotting with missing expected columns."""
        bad_df = pl.DataFrame({'feature': ['a'], 'counts': [1]}) # Missing required cols
        result = plot_feature_summary(bad_df, 'feat', 'target')
        self.assertEqual(result, "")
        mock_subplots.assert_not_called()


if __name__ == '__main__':
    unittest.main() 