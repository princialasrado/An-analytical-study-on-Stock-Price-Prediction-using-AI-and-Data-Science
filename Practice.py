import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from kivy.uix.popup import Popup
from kivy.core.window import Window

class StockApp(App):
    def build(self):
        # Set the app window size (optional)
        Window.size = (400, 600)

        self.model = load_model('stock_price_predictor.h5')
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Load company data (company name and ticker symbol)
        self.company_data = pd.read_csv("C:\\Users\\princ\\OneDrive\\Desktop\\Major projects\\Final.csv")  # Replace with your file path
        company_names = self.company_data['Company Name'].tolist()

        # Create the dropdown (Spinner) for company names
        self.company_spinner = Spinner(
            text='Select a Company',
            values=company_names,
            size_hint=(1, 0.2),
            font_size=18
        )
        layout.add_widget(self.company_spinner)

        # Create a custom button with background color and border
        predict_button = Button(
            text="Predict Stock Prices",
            font_size=18,
            size_hint=(1, 0.15),
            background_normal='',  # Removes default button background
            background_color=(0.1, 0.7, 0.3, 1),  # Custom background color (greenish)
            color=(1, 1, 1, 1),  # Text color (white)
            border=(2, 2, 2, 2)  # Adds a border (optional)
        )
        predict_button.bind(on_press=self.predict_prices)
        layout.add_widget(predict_button)

        return layout

    def predict_prices(self, instance):
        # Get selected company name from the dropdown
        selected_company = self.company_spinner.text
        if selected_company == 'Select a Company':
            self.show_popup("Error", "Please select a valid company.")
            return

        # Get the corresponding ticker symbol from the loaded company data
        company_row = self.company_data[self.company_data['Company Name'] == selected_company]
        if company_row.empty:
            self.show_popup("Error", "Company not found.")
            return
        
        company_ticker = company_row['Ticker Symbol'].values[0]

        try:
            # Fetch more data if needed (120 days, for instance)
            recent_data = yf.download(company_ticker, start=datetime.today() - timedelta(days=120), end=datetime.today())
            recent_open = recent_data['Open'].dropna().values.reshape(-1, 1)

            if len(recent_open) < 60:
                self.show_popup("Error", f"Not enough data to predict. Only {len(recent_open)} days of data found.")
                return

            # Scale the data using the trained scaler
            scaled_recent_open = self.scaler.fit_transform(recent_open)

            # Predict the next 30 days
            predicted_stock_prices = []
            input_sequence = scaled_recent_open[-60:].tolist()  # Last 60 days as input

            for day in range(30):
                # Convert input_sequence into a numpy array and reshape it for the LSTM model
                X_input = np.array(input_sequence[-60:])  # Take the last 60 timesteps for prediction
                X_input = np.reshape(X_input, (1, X_input.shape[0], 1))  # Reshape for LSTM input

                # Predict the next day's stock price
                predicted_price = self.model.predict(X_input)
                predicted_stock_prices.append(predicted_price[0, 0])  # Save the predicted value

                # Append the predicted value to the input sequence for the next day's prediction
                input_sequence.append([predicted_price[0, 0]])

            # Inverse transform the predicted prices back to the original scale
            predicted_stock_prices = np.array(predicted_stock_prices).reshape(-1, 1)
            predicted_stock_prices = self.scaler.inverse_transform(predicted_stock_prices)

            # Generate future dates for plotting (next 30 days)
            future_dates = pd.date_range(start=datetime.today(), periods=30).to_pydatetime()

            # Plot predicted future prices (next 30 days) as a **line graph**
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, predicted_stock_prices.flatten(), color='blue', marker='o', linestyle='-', label='Predicted Prices')

            plt.title(f'{selected_company} ({company_ticker}) Stock Price Prediction for Next 30 Days')
            plt.xlabel('Date')
            plt.ylabel('Predicted Stock Price')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.show_popup("Error", str(e))

    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(0.6, 0.4))
        popup.open()

if __name__ == "__main__":
    StockApp().run()
