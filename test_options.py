import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import matplotlib.pyplot as plt

def get_current_price(ticker_symbol):
    """Get the current price of a stock"""
    ticker = yf.Ticker(ticker_symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'].iloc[-1]

def get_weekly_options(ticker_symbol, weeks=6):
    """
    Get option chain data for a given ticker for the next specified number of weeks.
    
    Args:
        ticker_symbol (str): The stock ticker symbol
        weeks (int): Number of weeks to look ahead
        
    Returns:
        dict: Option chain data by expiration date
    """
    ticker = yf.Ticker(ticker_symbol)
    current_price = get_current_price(ticker_symbol)
    print(f"Current price of {ticker_symbol}: ${current_price:.2f}")
    
    # Get all available expiration dates
    try:
        expirations = ticker.options
        print(f"Available expiration dates for {ticker_symbol}: {expirations}")
    except Exception as e:
        print(f"Error fetching expiration dates: {e}")
        return {}
    
    # Calculate the dates for the next 'weeks' number of weeks
    today = datetime.now()
    target_dates = [(today + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, weeks+1)]
    
    print(f"Looking for expiration dates close to: {target_dates}")
    
    # Find the closest expiration dates to our target dates
    selected_expirations = []
    for target in target_dates:
        closest = None
        min_diff = float('inf')
        target_date = datetime.strptime(target, '%Y-%m-%d')
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            diff = abs((exp_date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest = exp
        
        if closest and closest not in selected_expirations:
            selected_expirations.append(closest)
    
    # Fetch option chains for selected expiration dates
    option_chains = {}
    for exp_date in selected_expirations:
        print(f"\nFetching option chain for expiration date: {exp_date}")
        try:
            # Adding delay to avoid rate limiting
            time.sleep(1)
            opt = ticker.option_chain(exp_date)
            
            print(f"Call options count: {len(opt.calls)}")
            print(f"Put options count: {len(opt.puts)}")
            
            # Extract key information for better readability
            calls_df = opt.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
            puts_df = opt.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
            
            # Add % distance from current price
            calls_df['distance_pct'] = ((calls_df['strike'] - current_price) / current_price) * 100
            puts_df['distance_pct'] = ((puts_df['strike'] - current_price) / current_price) * 100
            
            option_chains[exp_date] = {
                'calls': calls_df,
                'puts': puts_df,
                'days_to_expiration': (datetime.strptime(exp_date, '%Y-%m-%d') - today).days
            }
            
            # Print some sample data
            print("\nSample Call Options:")
            print(calls_df.head(3))
            print("\nSample Put Options:")
            print(puts_df.head(3))
            
            # Analyze potential credit spreads
            analyze_potential_credit_spreads(current_price, calls_df, puts_df, exp_date, option_chains[exp_date]['days_to_expiration'])
            
        except Exception as e:
            print(f"Error fetching option chain for {exp_date}: {e}")
    
    return option_chains

def analyze_potential_credit_spreads(current_price, calls_df, puts_df, exp_date, days_to_expiration):
    """Analyze and print potential credit spread opportunities"""
    print("\nPotential Credit Spread Opportunities:")
    print(f"Expiration: {exp_date} (Days to expiration: {days_to_expiration})")
    
    # For bull put spreads (bullish outlook)
    print("\nBull Put Spreads (Cash Secured Put alternatives):")
    # Filter for puts below current price (out of the money)
    otm_puts = puts_df[puts_df['strike'] < current_price].copy()
    if not otm_puts.empty:
        # Sort by strike price descending
        otm_puts = otm_puts.sort_values('strike', ascending=False)
        
        # Display the 3 closest to money puts with decent volume
        liquid_otm_puts = otm_puts[otm_puts['volume'] > 10].head(3)
        if not liquid_otm_puts.empty:
            for _, short_put in liquid_otm_puts.iterrows():
                # Find a long put with lower strike to define the spread
                potential_long_puts = otm_puts[otm_puts['strike'] < short_put['strike']]
                if not potential_long_puts.empty:
                    long_put = potential_long_puts.iloc[0]
                    
                    # Calculate credit, max loss, and return
                    width = short_put['strike'] - long_put['strike']
                    credit = max(0, short_put['bid'] - long_put['ask'])
                    max_loss = width - credit
                    if max_loss > 0:
                        roi = (credit / max_loss) * 100
                        annualized_roi = roi * (365 / days_to_expiration)
                        
                        print(f"Short {short_put['strike']} Put / Long {long_put['strike']} Put:")
                        print(f"  Credit: ${credit:.2f}, Max Loss: ${max_loss:.2f}")
                        print(f"  Potential Return: {roi:.2f}% ({annualized_roi:.2f}% annualized)")
                        print(f"  Short Put distance from current price: {short_put['distance_pct']:.2f}%")
        else:
            print("  No liquid OTM puts found for bull put spreads")
    else:
        print("  No OTM puts available")
        
    # For bear call spreads (bearish outlook)
    print("\nBear Call Spreads (Selling covered calls alternative):")
    # Filter for calls above current price (out of the money)
    otm_calls = calls_df[calls_df['strike'] > current_price].copy()
    if not otm_calls.empty:
        # Sort by strike price ascending
        otm_calls = otm_calls.sort_values('strike')
        
        # Display the 3 closest to money calls with decent volume
        liquid_otm_calls = otm_calls[otm_calls['volume'] > 10].head(3)
        if not liquid_otm_calls.empty:
            for _, short_call in liquid_otm_calls.iterrows():
                # Find a long call with higher strike to define the spread
                potential_long_calls = otm_calls[otm_calls['strike'] > short_call['strike']]
                if not potential_long_calls.empty:
                    long_call = potential_long_calls.iloc[0]
                    
                    # Calculate credit, max loss, and return
                    width = long_call['strike'] - short_call['strike']
                    credit = max(0, short_call['bid'] - long_call['ask'])
                    max_loss = width - credit
                    if max_loss > 0:
                        roi = (credit / max_loss) * 100
                        annualized_roi = roi * (365 / days_to_expiration)
                        
                        print(f"Short {short_call['strike']} Call / Long {long_call['strike']} Call:")
                        print(f"  Credit: ${credit:.2f}, Max Loss: ${max_loss:.2f}")
                        print(f"  Potential Return: {roi:.2f}% ({annualized_roi:.2f}% annualized)")
                        print(f"  Short Call distance from current price: {short_call['distance_pct']:.2f}%")
        else:
            print("  No liquid OTM calls found for bear call spreads")
    else:
        print("  No OTM calls available")

def analyze_option_chain_metrics(option_chains):
    """Analyze and display metrics about the option chains"""
    print("\n=== OPTION CHAIN ANALYSIS ===")
    
    # Extract expirations in chronological order
    expirations = sorted(option_chains.keys())
    
    # Track metrics across expirations
    call_iv_by_exp = {}
    put_iv_by_exp = {}
    volume_by_exp = {}
    
    for exp in expirations:
        chain_data = option_chains[exp]
        days = chain_data['days_to_expiration']
        
        calls = chain_data['calls']
        puts = chain_data['puts']
        
        # Calculate average IV for ATM options (within 5% of current price)
        atm_calls = calls[abs(calls['distance_pct']) < 5]
        atm_puts = puts[abs(puts['distance_pct']) < 5]
        
        avg_call_iv = atm_calls['impliedVolatility'].mean() * 100 if not atm_calls.empty else 0
        avg_put_iv = atm_puts['impliedVolatility'].mean() * 100 if not atm_puts.empty else 0
        
        call_iv_by_exp[exp] = avg_call_iv
        put_iv_by_exp[exp] = avg_put_iv
        
        # Calculate total volume
        total_volume = calls['volume'].sum() + puts['volume'].sum()
        volume_by_exp[exp] = total_volume
        
        print(f"\nExpiration: {exp} (Days: {days})")
        print(f"  Average ATM Call IV: {avg_call_iv:.2f}%")
        print(f"  Average ATM Put IV: {avg_put_iv:.2f}%")
        print(f"  Total Volume: {total_volume:.0f}")
        
        # Check for skew (difference between put and call IV)
        skew = avg_put_iv - avg_call_iv
        print(f"  IV Skew (Put - Call): {skew:.2f}%")
        if skew > 3:
            print("  Market shows defensive positioning (puts more expensive than calls)")
        elif skew < -3:
            print("  Market shows aggressive positioning (calls more expensive than puts)")
        else:
            print("  Market shows neutral positioning")
    
    # Try to identify the best expiration for credit spreads based on IV and days
    best_bull_exp = max(expirations, key=lambda x: put_iv_by_exp[x] / (option_chains[x]['days_to_expiration'] + 1))
    best_bear_exp = max(expirations, key=lambda x: call_iv_by_exp[x] / (option_chains[x]['days_to_expiration'] + 1))
    
    print("\nBest Expirations for Credit Spreads:")
    print(f"  Bull Put Spreads: {best_bull_exp} (High put IV relative to timeframe)")
    print(f"  Bear Call Spreads: {best_bear_exp} (High call IV relative to timeframe)")

# Execute for TSLA
if __name__ == "__main__":
    print("Fetching TSLA option chains for the next 6 weeks...")
    options_data = get_weekly_options("TSLA", weeks=6)
    
    if options_data:
        # Count total options
        total_calls = sum(len(data['calls']) for data in options_data.values())
        total_puts = sum(len(data['puts']) for data in options_data.values())
        
        print(f"\nSummary:")
        print(f"Retrieved option data for {len(options_data)} expiration dates")
        print(f"Total call options: {total_calls}")
        print(f"Total put options: {total_puts}")
        
        # Analyze metrics across all option chains
        analyze_option_chain_metrics(options_data)
    else:
        print("Failed to retrieve option data") 