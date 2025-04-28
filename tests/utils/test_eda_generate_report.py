# Tests for generate_eda_report function.

import unittest
from unittest.mock import patch, MagicMock, call, mock_open
import polars as pl
import datetime

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_generate_report import generate_eda_report
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_generate_report import generate_eda_report

# Use qualified names for patching if they are imported directly in the module
@patch('utils.eda_generate_report.plot_target_distribution')
@patch('utils.eda_generate_report.calculate_correlations')
@patch('utils.eda_generate_report.plot_correlation_heatmap')
@patch('utils.eda_generate_report.plot_target_correlation_bar')
@patch('utils.eda_generate_report.analyze_target_by_feature')
@patch('utils.eda_generate_report.plot_feature_summary')
@patch("builtins.open", new_callable=mock_open) # Mock file writing
class TestGenerateEDAReport(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            'target': [1.0, 2.0, 1.5, 3.0],
            'feat1': ['A', 'B', 'A', 'B'],
            'feat2': [10, 20, 15, 25]
        })
        self.target_col = 'target'
        self.output_path = "/fake/report.html"
        
        # Sample return values for mocks
        self.mock_target_dist_b64 = "TARGET_DIST_B64"
        self.mock_feat_corr = pl.DataFrame({'feature': ['feat2'], 'feat2': [1.0]})
        self.mock_target_corr = pl.DataFrame({'feature': ['feat2'], 'target_correlation': [0.9]})
        self.mock_heatmap_b64 = "HEATMAP_B64"
        self.mock_target_bar_b64 = "TARGET_BAR_B64"
        self.mock_summary_feat1 = pl.DataFrame({'feature_value': ['A', 'B'], 'count': [2,2], 'target_mean': [1.25, 2.5], 'target_median': [1.25, 2.5]})
        self.mock_summary_feat2 = pl.DataFrame({'feature_value': ['q1', 'q2'], 'count': [2,2], 'target_mean': [1.25, 2.5], 'target_median': [1.25, 2.5]})
        self.mock_summary_plot_b64 = "SUMMARY_PLOT_B64"

    def test_report_generation_all_success(self, mock_file_open, mock_plot_summary, mock_analyze, 
                                           mock_plot_target_bar, mock_plot_heatmap, mock_calc_corr, 
                                           mock_plot_target_dist):
        """Test successful report generation with all components."""
        # Configure mock returns
        mock_plot_target_dist.return_value = self.mock_target_dist_b64
        mock_calc_corr.return_value = (self.mock_feat_corr, self.mock_target_corr)
        mock_plot_heatmap.return_value = self.mock_heatmap_b64
        mock_plot_target_bar.return_value = self.mock_target_bar_b64
        mock_analyze.side_effect = [self.mock_summary_feat1, self.mock_summary_feat2] # One per feature
        mock_plot_summary.return_value = self.mock_summary_plot_b64
        
        generate_eda_report(self.df, self.target_col, self.output_path, features_to_analyze=['feat1', 'feat2'])

        # Check mocks were called
        mock_plot_target_dist.assert_called_once_with(self.df, self.target_col)
        mock_calc_corr.assert_called_once_with(self.df, self.target_col)
        mock_plot_heatmap.assert_called_once_with(self.mock_feat_corr, max_size_for_annot=50)
        mock_plot_target_bar.assert_called_once_with(self.mock_target_corr, self.target_col, max_features_to_plot=50)
        self.assertEqual(mock_analyze.call_count, 2)
        self.assertEqual(mock_plot_summary.call_count, 2)
        mock_file_open.assert_called_once_with(self.output_path, 'w', encoding='utf-8')
        
        # Check content written to file (basic checks)
        handle = mock_file_open()
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)
        
        self.assertIn("<title>Exploratory Data Analysis Report</title>", written_content)
        self.assertIn(self.mock_target_dist_b64, written_content)
        self.assertIn(self.mock_heatmap_b64, written_content)
        self.assertIn(self.mock_target_bar_b64, written_content)
        self.assertIn(self.mock_summary_plot_b64, written_content) # Should appear twice
        self.assertIn("Analysis for Feature: feat1", written_content)
        self.assertIn("Analysis for Feature: feat2", written_content)
        self.assertIn("<table", written_content) # Check tables are included

    def test_report_analyze_all_features(self, mock_file_open, mock_plot_summary, mock_analyze, 
                                          mock_plot_target_bar, mock_plot_heatmap, mock_calc_corr, 
                                          mock_plot_target_dist):
        """Test report generation analyzing all suitable features when None is passed."""
        # Mock returns
        mock_plot_target_dist.return_value = "b64"
        mock_calc_corr.return_value = (None, None) # Simplify
        mock_analyze.side_effect = [self.mock_summary_feat1, self.mock_summary_feat2] 
        mock_plot_summary.return_value = "b64"

        generate_eda_report(self.df, self.target_col, self.output_path, features_to_analyze=None)
        
        # Should analyze 'feat1' and 'feat2'
        self.assertEqual(mock_analyze.call_count, 2)
        calls = [call(self.df, target_col='target', feature_col='feat1', max_categories=20, num_bins=10), 
                 call(self.df, target_col='target', feature_col='feat2', max_categories=20, num_bins=10)]
        mock_analyze.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_plot_summary.call_count, 2)
        mock_file_open.assert_called_once_with(self.output_path, 'w', encoding='utf-8')

    def test_report_skip_non_numeric_target_analysis(self, mock_file_open, mock_plot_summary, mock_analyze, 
                                                   mock_plot_target_bar, mock_plot_heatmap, mock_calc_corr, 
                                                   mock_plot_target_dist):
        """Test that detailed feature analysis is skipped if target is non-numeric."""
        df_cat_target = self.df.with_columns(pl.col('target').cast(str))
        
        # Mock returns (calc_corr will return None for target_corr)
        mock_plot_target_dist.return_value = "b64" # This might still work if cast
        mock_calc_corr.return_value = (self.mock_feat_corr, None)
        mock_plot_heatmap.return_value = "b64"
        
        generate_eda_report(df_cat_target, self.target_col, self.output_path)

        mock_analyze.assert_not_called() # Should not be called
        mock_plot_summary.assert_not_called()
        
        handle = mock_file_open()
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)
        self.assertIn("Target column 'target' is not numeric. Skipping detailed feature analysis", written_content)
        self.assertNotIn("Analysis for Feature: feat1", written_content)

    def test_report_handle_analysis_errors(self, mock_file_open, mock_plot_summary, mock_analyze, 
                                            mock_plot_target_bar, mock_plot_heatmap, mock_calc_corr, 
                                            mock_plot_target_dist):
        """Test report generation when some analyses fail."""
        # Mock returns: Target dist fails, correlations OK, feat1 analyze OK, feat2 analyze fails
        mock_plot_target_dist.side_effect = ValueError("Target plot failed")
        mock_calc_corr.return_value = (self.mock_feat_corr, self.mock_target_corr)
        mock_plot_heatmap.return_value = self.mock_heatmap_b64
        mock_plot_target_bar.return_value = self.mock_target_bar_b64
        mock_analyze.side_effect = [self.mock_summary_feat1, ValueError("Analyze feat2 failed")]
        mock_plot_summary.return_value = self.mock_summary_plot_b64 # Only called for feat1

        generate_eda_report(self.df, self.target_col, self.output_path, features_to_analyze=['feat1', 'feat2'])

        mock_analyze.assert_has_calls([
            call(self.df, target_col='target', feature_col='feat1', max_categories=20, num_bins=10),
            call(self.df, target_col='target', feature_col='feat2', max_categories=20, num_bins=10)
        ])
        mock_plot_summary.assert_called_once() # Only called for the successful feat1 analysis
        
        handle = mock_file_open()
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)
        
        self.assertIn("Error generating target distribution plot: Target plot failed", written_content)
        self.assertIn("Analysis for Feature: feat1", written_content)
        self.assertIn(self.mock_summary_plot_b64, written_content) # Plot for feat1
        self.assertIn("Analysis for Feature: feat2", written_content)
        self.assertIn("Error during analysis of feature 'feat2': Analyze feat2 failed", written_content)

    def test_report_handle_missing_feature_analyze(self, mock_file_open, mock_plot_summary, mock_analyze, 
                                                   mock_plot_target_bar, mock_plot_heatmap, mock_calc_corr, 
                                                   mock_plot_target_dist):
        """Test skipping a feature specified for analysis if it doesn't exist."""
        # Mock returns 
        mock_plot_target_dist.return_value = "b64"
        mock_calc_corr.return_value = (None, None) # Simplify
        mock_analyze.side_effect = [self.mock_summary_feat1] # Only called for feat1
        mock_plot_summary.return_value = "b64"

        generate_eda_report(self.df, self.target_col, self.output_path, features_to_analyze=['feat1', 'missing_feat'])
        
        mock_analyze.assert_called_once_with(self.df, target_col='target', feature_col='feat1', max_categories=20, num_bins=10)
        mock_plot_summary.assert_called_once()

        handle = mock_file_open()
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)
        self.assertIn("Analysis for Feature: feat1", written_content)
        self.assertNotIn("Analysis for Feature: missing_feat", written_content) # Should be skipped


if __name__ == '__main__':
    unittest.main() 