# Black-Scholes Calculator Application

A professional-grade Option Calculator built with **FastAPI** (Backend) and **Streamlit** (Frontend).
It calculates the theoretical value of options using the Black-Scholes model and compares them with real-time market data from Yahoo Finance.

## Features

- **Real-time Data**: Fetches stock price, risk-free rate (TNX), and option chains.
- **Black-Scholes Model**: accurate pricing for Calls and Puts.
- **Greeks**: Delta, Gamma, Theta, Vega, Rho.
- **Comparison**: Compare Theoretical Value vs Actual Market Price.
- **Visualizations**: Interactive Payoff Diagrams.

## Prerequisites

- Python 3.9+
- Internet connection (for market data)

## Installation

1. Clone or download this repository.
2. Navigate to the project folder:

   ```bash
   cd BlackScholesApp
   ```

3. Set up the backend environment:

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the App

You need to run **two** processes in separate terminals.

### 1. Start the Backend API

From the `backend` directory (with venv activated):

```bash
uvicorn main:app --reload
```

*API will run at <http://127.0.0.1:8000>*

### 2. Start the Frontend

Open a new terminal, navigate to the project root, activate the venv, and run:

```bash
# From BlackScholesApp root
source backend/venv/bin/activate
streamlit run frontend/app.py
```

*App will open in your browser at <http://localhost:8501>*

## Usage

1. Enter a valid Ticker (e.g., `AAPL`, `SPY`).
2. Select an Expiration Date from the dropdown.
3. Choose Call or Put.
4. Select a Strike Price (closest to money selected by default).
5. Modify Volatility if desired (defaults to Implied Volatility if available, or Historical).
6. Click **Calculate** to see the comparison and Greeks.
