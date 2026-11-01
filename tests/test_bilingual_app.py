"""Run from the repository root: python -m unittest discover -s tests."""
import unittest

from streamlit.testing.v1 import AppTest


class BilingualAppTest(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertEqual([e.message for e in app.exception], [])

    def test_languages_pages_and_model_state(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assert_healthy(app)
        self.assertEqual(app.title[0].value, "Sleep, Sport & Stress Dashboard")
        pages = [
            ("utils/dashboard.py", "Dashboard für Schlaf, Sport & Stress", "Sleep, Sport & Stress Dashboard"),
            ("views/1_Daten_Exploration.py", "Datenexploration", "Data Exploration"),
            ("views/2_Visualisierung.py", "Visualisierung", "Visualization"),
            ("views/4_ML_Ridge_Regression.py", "HRV-Vorhersage mit Ridge-Regression", "HRV Prediction with Ridge Regression"),
            ("views/5_ML_Logistic_Regression.py", "HRV-Stressklassifikation mit logistischer Regression", "HRV Stress Classification with Logistic Regression"),
        ]
        for language, title_index in [("de", 1), ("en", 2)]:
            app.selectbox(key="language").select(language).run()
            for page in pages:
                app.switch_page(page[0]).run(timeout=30)
                self.assert_healthy(app)
                self.assertEqual(app.title[0].value, page[title_index])
                self.assertEqual(app.selectbox(key="language").value, language)
                if "dashboard" in page[0] or "Exploration" in page[0]:
                    self.assertIn("Datum" if language == "de" else "Date", app.dataframe[0].value.columns)

        # A language change must retain both trained models and current inputs.
        for page, prefix, model_key in [
            (pages[3][0], "ridge", "ridge_model"),
            (pages[4][0], "logreg", "logreg_model"),
        ]:
            app.switch_page(page).run()
            app.button[0].click().run(timeout=30)
            self.assert_healthy(app)
            model = app.session_state[model_key]
            app.number_input(key=prefix + "_sleep_hours").set_value(8.5).run()
            app.slider(key=prefix + "_sleep_quality").set_value(75.0).run()
            for language in ["de", "en"]:
                app.selectbox(key="language").select(language).run()
                self.assert_healthy(app)
                self.assertIs(app.session_state[model_key], model)
                self.assertEqual(app.number_input(key=prefix + "_sleep_hours").value, 8.5)
                self.assertEqual(app.slider(key=prefix + "_sleep_quality").value, 75.0)
                app.button[1].click().run(timeout=30)
                self.assert_healthy(app)
                messages = [item.value for item in [*app.success, *app.error]]
                self.assertTrue(any("vorhergesagt" in x.lower() if language == "de" else "predicted" in x.lower() for x in messages))
        # Navigating back must keep the ridge evaluation separate from classification.
        app.switch_page(pages[3][0]).run()
        self.assert_healthy(app)
        self.assertGreater(app.session_state["ridge_y_test"].mean(), 1)
        self.assertLessEqual(app.session_state["logreg_y_test"].max(), 1)


if __name__ == "__main__":
    unittest.main()
