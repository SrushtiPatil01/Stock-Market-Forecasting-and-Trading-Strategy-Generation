import pandas as pd
import streamlit as st


class FeatureDatabase:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data = None

    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            return True
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return False

    def display_data(self, num_rows=None):
        """Display the data in Streamlit"""
        if self.data is None:
            st.warning("No data loaded. Please call load_data() first.")
            return

        if num_rows:
            st.dataframe(self.data.head(num_rows), height=10000)
        else:
            st.dataframe(self.data)

    def get_data(self):
        """Return the DataFrame for external use"""
        return self.data
