import unittest
import numpy as np
import polars as pl
from polars.testing import assert_series_equal
from sklearn import metrics
from utils.calculate_metric import calculate_metric

class TestCalculateMetric(unittest.TestCase):

    # --- Classification Setups ---
    def setUpClass_classification(cls):
        cls.y_true_clf = np.array([0, 1, 0, 1, 0, 0, 1])
        cls.y_pred_clf_perfect = np.array([0, 1, 0, 1, 0, 0, 1])
        cls.y_pred_clf_imperfect = np.array([0, 0, 1, 1, 0, 1, 0])
        cls.y_pred_clf_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.7]) # Probs for class 1
        cls.y_true_clf_multi = np.array([0, 1, 2, 0, 1, 2])
        cls.y_pred_clf_multi = np.array([0, 2, 1, 0, 0, 1])

    # --- Regression Setups ---
    def setUpClass_regression(cls):
        cls.y_true_reg = np.array([1.0, 2.5, 3.0, 4.2, 5.0])
        cls.y_pred_reg_perfect = np.array([1.0, 2.5, 3.0, 4.2, 5.0])
        cls.y_pred_reg_imperfect = np.array([1.2, 2.3, 3.5, 4.0, 5.5])
        # For RMSLE - ensure no negative values
        cls.y_true_reg_pos = np.array([1, 2, 3, 4, 5])
        cls.y_pred_reg_pos = np.array([1.1, 1.9, 3.2, 3.8, 5.3])

    def setUp(self):
        # Run setup methods for different test types
        self.setUpClass_classification()
        self.setUpClass_regression()

    # --- Classification Tests ---
    def test_accuracy(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_perfect, 'accuracy'), 1.0)
        # Imperfect: Correct = 3/7
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_imperfect, 'accuracy'), 3/7)

    def test_precision_recall_f1_binary(self):
        # Perfect
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_perfect, 'precision'), 1.0)
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_perfect, 'recall'), 1.0)
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_perfect, 'f1'), 1.0)
        # Imperfect: TP=1, FP=2, FN=2 -> Prec=1/3, Rec=1/3, F1=2*(1/3*1/3)/(1/3+1/3)=1/3
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_imperfect, 'precision'), 1/3)
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_imperfect, 'recall'), 1/3)
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_imperfect, 'f1'), 1/3)

    def test_roc_auc(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_proba, 'roc_auc'),
                               metrics.roc_auc_score(self.y_true_clf, self.y_pred_clf_proba))
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_proba, 'AUC'), # Case-insensitive
                               metrics.roc_auc_score(self.y_true_clf, self.y_pred_clf_proba))

    def test_logloss(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_proba, 'logloss'),
                               metrics.log_loss(self.y_true_clf, self.y_pred_clf_proba))

    def test_balanced_accuracy(self):
         self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_imperfect, 'balanced_accuracy'),
                                metrics.balanced_accuracy_score(self.y_true_clf, self.y_pred_clf_imperfect))

    def test_f1_micro_macro(self):
        # Use multiclass data
        f1_micro = calculate_metric(self.y_true_clf_multi, self.y_pred_clf_multi, 'f1', average='micro')
        f1_macro = calculate_metric(self.y_true_clf_multi, self.y_pred_clf_multi, 'f1', average='macro')
        self.assertAlmostEqual(f1_micro, metrics.f1_score(self.y_true_clf_multi, self.y_pred_clf_multi, average='micro'))
        self.assertAlmostEqual(f1_macro, metrics.f1_score(self.y_true_clf_multi, self.y_pred_clf_multi, average='macro'))

    # --- Regression Tests ---
    def test_mse(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'mse'), 0.0)
        expected_mse = np.mean((self.y_true_reg - self.y_pred_reg_imperfect)**2)
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_imperfect, 'mse'), expected_mse)

    def test_rmse(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'rmse'), 0.0)
        expected_rmse = np.sqrt(np.mean((self.y_true_reg - self.y_pred_reg_imperfect)**2))
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_imperfect, 'rmse'), expected_rmse)

    def test_mae(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'mae'), 0.0)
        expected_mae = np.mean(np.abs(self.y_true_reg - self.y_pred_reg_imperfect))
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_imperfect, 'mae'), expected_mae)

    def test_r2(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'r2'), 1.0)
        expected_r2 = metrics.r2_score(self.y_true_reg, self.y_pred_reg_imperfect)
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_imperfect, 'r2'), expected_r2)

    def test_rmsle(self):
        # Test with positive values only
        self.assertAlmostEqual(calculate_metric(self.y_true_reg_pos, self.y_pred_reg_pos, 'rmsle'),
                              np.sqrt(metrics.mean_squared_log_error(self.y_true_reg_pos, self.y_pred_reg_pos)))
        # Test case where it should fail (negative values)
        y_true_neg = np.array([-1, 2])
        y_pred_neg = np.array([1, 3])
        with self.assertRaises(ValueError): # MSLE requires non-negative values
             calculate_metric(y_true_neg, y_pred_neg, 'rmsle')

    def test_mape(self):
         self.assertAlmostEqual(calculate_metric(self.y_true_reg_pos, self.y_pred_reg_pos, 'mape'),
                                metrics.mean_absolute_percentage_error(self.y_true_reg_pos, self.y_pred_reg_pos))

    # --- General Tests ---
    def test_polars_series_input(self):
        """Test calculation with Polars Series inputs."""
        y_true_pl = pl.Series("true", self.y_true_reg)
        y_pred_pl = pl.Series("pred", self.y_pred_reg_imperfect)
        expected_mae = np.mean(np.abs(self.y_true_reg - self.y_pred_reg_imperfect))
        self.assertAlmostEqual(calculate_metric(y_true_pl, y_pred_pl, 'mae'), expected_mae)

        # Test classification metric too
        y_true_clf_pl = pl.Series("true_clf", self.y_true_clf)
        y_pred_clf_pl = pl.Series("pred_clf", self.y_pred_clf_imperfect)
        self.assertAlmostEqual(calculate_metric(y_true_clf_pl, y_pred_clf_pl, 'accuracy'), 3/7)

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'invalid_metric_name')

    def test_metric_case_insensitivity(self):
        self.assertAlmostEqual(calculate_metric(self.y_true_reg, self.y_pred_reg_perfect, 'MSE'), 0.0)
        self.assertAlmostEqual(calculate_metric(self.y_true_clf, self.y_pred_clf_perfect, 'Accuracy'), 1.0)

if __name__ == '__main__':
    unittest.main() 