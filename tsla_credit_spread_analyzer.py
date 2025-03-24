import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import time
import os

class TeslaCreditSpreadAnalyzer:
    """
    Analyzes Tesla options data to find optimal credit spread opportunities.
    This class helps identify potentially profitable credit spread setups based on
    technical indicators, implied volatility, and option chain analysis.
    """
    
    def __init__(self):
        self.ticker_symbol = "TSLA"
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.current_price = self._get_current_price()
        self.technical_indicators = {}
        self.option_chains = {}
        
    def _get_current_price(self):
        """Get the current price of Tesla stock"""
        todays_data = self.ticker.history(period='1d')
        return todays_data['Close'].iloc[-1]
    
    def load_history(self, period="6mo", interval="1d"):
        """Load historical price data and calculate technical indicators"""
        print(f"Loading {self.ticker_symbol} historical data for {period}...")
        self.history = self.ticker.history(period=period, interval=interval)
        print(f"Loaded {len(self.history)} data points")
        
        # Calculate technical indicators
        self._calculate_technical_indicators()
        return self.history
    
    def _calculate_technical_indicators(self):
        """Calculate relevant technical indicators for trading decisions"""
        if self.history is None or len(self.history) == 0:
            print("No historical data to calculate indicators")
            return
        
        # Calculate 20-day and 50-day moving averages
        self.history['MA20'] = self.history['Close'].rolling(window=20).mean()
        self.history['MA50'] = self.history['Close'].rolling(window=50).mean()
        
        # Calculate RSI (14-period)
        delta = self.history['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        self.history['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands (20-day, 2 standard deviations)
        self.history['BB_middle'] = self.history['Close'].rolling(window=20).mean()
        std_dev = self.history['Close'].rolling(window=20).std()
        self.history['BB_upper'] = self.history['BB_middle'] + (std_dev * 2)
        self.history['BB_lower'] = self.history['BB_middle'] - (std_dev * 2)
        
        # Store last values for analysis
        last_row = self.history.iloc[-1]
        self.technical_indicators = {
            'price': last_row['Close'],
            'ma20': last_row['MA20'],
            'ma50': last_row['MA50'],
            'rsi': last_row['RSI'],
            'bb_upper': last_row['BB_upper'],
            'bb_middle': last_row['BB_middle'],
            'bb_lower': last_row['BB_lower'],
            'price_to_ma20': last_row['Close'] / last_row['MA20'] - 1,
            'price_to_ma50': last_row['Close'] / last_row['MA50'] - 1,
            'bb_width': (last_row['BB_upper'] - last_row['BB_lower']) / last_row['BB_middle'],
        }
        
        print("\nTechnical Indicators:")
        print(f"Current Price: ${self.technical_indicators['price']:.2f}")
        print(f"20-day MA: ${self.technical_indicators['ma20']:.2f} ({self.technical_indicators['price_to_ma20']*100:.2f}%)")
        print(f"50-day MA: ${self.technical_indicators['ma50']:.2f} ({self.technical_indicators['price_to_ma50']*100:.2f}%)")
        print(f"RSI (14): {self.technical_indicators['rsi']:.2f}")
        print(f"Bollinger Band Width: {self.technical_indicators['bb_width']*100:.2f}%")
    
    def analyze_market_trend(self):
        """Analyze the current market trend for TSLA"""
        if not self.technical_indicators:
            print("Technical indicators not calculated. Run load_history() first.")
            return "UNKNOWN"
        
        ti = self.technical_indicators
        
        # Define trend criteria
        trend = "NEUTRAL"
        trend_strength = 0
        
        # Bullish conditions
        if ti['price'] > ti['ma20'] > ti['ma50']:
            trend = "BULLISH"
            trend_strength += 1
        if ti['price_to_ma20'] > 0.03:  # Price 3% above 20-day MA
            trend_strength += 1
        if ti['rsi'] > 50 and ti['rsi'] < 70:
            trend_strength += 1
        if ti['price'] > ti['bb_middle']:
            trend_strength += 1
            
        # Bearish conditions
        if ti['price'] < ti['ma20'] < ti['ma50']:
            trend = "BEARISH"
            trend_strength += 1
        if ti['price_to_ma20'] < -0.03:  # Price 3% below 20-day MA
            trend_strength += 1
        if ti['rsi'] < 50 and ti['rsi'] > 30:
            trend_strength += 1
        if ti['price'] < ti['bb_middle']:
            trend_strength += 1
            
        # Overbought/Oversold conditions
        if ti['rsi'] > 70:
            trend = "OVERBOUGHT"
        if ti['rsi'] < 30:
            trend = "OVERSOLD"
            
        # High volatility
        if ti['bb_width'] > 0.2:  # BB width over 20%
            trend += "_VOLATILE"
            
        # Combine trend and strength
        result = f"{trend} (Strength: {trend_strength}/4)"
        print(f"\nMarket Trend Analysis: {result}")
        return result
    
    def load_option_chains(self, weeks=6):
        """
        Load option chains for the next specified number of weeks
        """
        print(f"\nLoading option chains for the next {weeks} weeks...")
        
        # Get all available expiration dates
        try:
            expirations = self.ticker.options
            print(f"Available expiration dates: {expirations}")
        except Exception as e:
            print(f"Error fetching expiration dates: {e}")
            return {}
        
        # Calculate the dates for the next 'weeks' number of weeks
        today = datetime.now()
        target_dates = [(today + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, weeks+1)]
        
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
        for exp_date in selected_expirations:
            print(f"\nFetching option chain for expiration date: {exp_date}")
            try:
                # Adding delay to avoid rate limiting
                time.sleep(1)
                opt = self.ticker.option_chain(exp_date)
                
                calls_df = opt.calls.copy()
                puts_df = opt.puts.copy()
                
                # Add % distance from current price
                calls_df['distance_pct'] = ((calls_df['strike'] - self.current_price) / self.current_price) * 100
                puts_df['distance_pct'] = ((puts_df['strike'] - self.current_price) / self.current_price) * 100
                
                # Calculate option-specific metrics
                calls_df['bid_ask_spread'] = calls_df['ask'] - calls_df['bid']
                puts_df['bid_ask_spread'] = puts_df['ask'] - puts_df['bid']
                
                # Filter for better spreads where data is available
                valid_calls = calls_df[(calls_df['bid'] > 0) & (calls_df['ask'] > 0)]
                valid_puts = puts_df[(puts_df['bid'] > 0) & (puts_df['ask'] > 0)]
                
                if not valid_calls.empty:
                    calls_df['bid_ask_ratio'] = calls_df['bid'] / calls_df['ask']
                if not valid_puts.empty:
                    puts_df['bid_ask_ratio'] = puts_df['bid'] / puts_df['ask']
                
                days_to_expiration = (datetime.strptime(exp_date, '%Y-%m-%d') - today).days
                
                self.option_chains[exp_date] = {
                    'calls': calls_df,
                    'puts': puts_df,
                    'days_to_expiration': days_to_expiration
                }
                
                print(f"  Days to expiration: {days_to_expiration}")
                print(f"  Call options: {len(calls_df)}")
                print(f"  Put options: {len(puts_df)}")
                
            except Exception as e:
                print(f"Error fetching option chain for {exp_date}: {e}")
        
        return self.option_chains
    
    def find_credit_spread_opportunities(self, min_credit=0.10, max_risk_reward_ratio=10):
        """
        Find potential credit spread opportunities based on current market conditions
        
        Args:
            min_credit: Minimum credit (premium) to collect for a spread
            max_risk_reward_ratio: Maximum risk/reward ratio to consider
        """
        if not self.option_chains:
            print("Option chains not loaded. Run load_option_chains() first.")
            return
        
        # Determine market trend to select strategy focus
        market_trend = self.analyze_market_trend()
        bull_put_score_multiplier = 1.0
        bear_call_score_multiplier = 1.0
        
        # Adjust strategy based on market trend
        if "BULLISH" in market_trend:
            bull_put_score_multiplier = 1.5
            print("\nFavoring bull put spreads due to bullish trend")
        elif "BEARISH" in market_trend:
            bear_call_score_multiplier = 1.5
            print("\nFavoring bear call spreads due to bearish trend")
        elif "OVERBOUGHT" in market_trend:
            bear_call_score_multiplier = 1.3
            print("\nSlightly favoring bear call spreads due to overbought conditions")
        elif "OVERSOLD" in market_trend:
            bull_put_score_multiplier = 1.3
            print("\nSlightly favoring bull put spreads due to oversold conditions")
        
        # Store all opportunities for comparison
        bull_put_opportunities = []
        bear_call_opportunities = []
        
        for exp_date, chain_data in self.option_chains.items():
            days = chain_data['days_to_expiration']
            calls = chain_data['calls']
            puts = chain_data['puts']
            
            # Skip expiration dates that are too close or too far
            if days < 7 or days > 60:
                continue
                
            print(f"\nAnalyzing spreads for expiration: {exp_date} (Days: {days})")
            
            # Calculate reasonable strike ranges based on historical volatility
            if 'bb_width' in self.technical_indicators:
                volatility_factor = self.technical_indicators['bb_width'] * 3  # Use BB width as volatility estimate
            else:
                volatility_factor = 0.15  # Default 15% range if no BB data
                
            # Adjust strike range based on days to expiration
            time_factor = min(days / 30, 2)  # Cap at 2x for longer dates
            range_percent = volatility_factor * time_factor * 100
            
            # Define potential strike ranges
            low_strike = self.current_price * (1 - range_percent/100)
            high_strike = self.current_price * (1 + range_percent/100)
            
            print(f"  Strike range: ${low_strike:.2f} to ${high_strike:.2f} (±{range_percent:.1f}%)")
            
            # Find bull put spread opportunities
            otm_puts = puts[puts['strike'] < self.current_price].copy()
            if not otm_puts.empty:
                # Sort by strike descending (closest to money first)
                otm_puts = otm_puts.sort_values('strike', ascending=False)
                
                # Take 10 closest strikes to current price for short put
                for short_idx, short_put in otm_puts.head(10).iterrows():
                    # Find potential long puts with lower strikes
                    long_puts = otm_puts[otm_puts['strike'] < short_put['strike']]
                    
                    if long_puts.empty:
                        continue
                    
                    # Get the next strike down for the long put
                    long_put = long_puts.iloc[0]
                    
                    # Calculate spread metrics
                    width = short_put['strike'] - long_put['strike']
                    
                    # Check if strikes are valid
                    if width < 1:
                        continue
                        
                    # Calculate potential credit
                    # If bid/ask are zero, use last price as a fallback
                    short_price = short_put['bid'] if short_put['bid'] > 0 else short_put['lastPrice'] * 0.9
                    long_price = long_put['ask'] if long_put['ask'] > 0 else long_put['lastPrice'] * 1.1
                    
                    credit = max(0.01, short_price - long_price)  # Ensure minimum credit
                    
                    if credit < min_credit:
                        continue
                        
                    max_loss = width - credit
                    risk_reward_ratio = max_loss / credit
                    
                    if risk_reward_ratio > max_risk_reward_ratio:
                        continue
                        
                    roi = (credit / max_loss) * 100
                    annualized_roi = roi * (365 / days)
                    
                    # Calculate distance from current price as percentage
                    distance_pct = ((short_put['strike'] - self.current_price) / self.current_price) * 100
                    
                    # Higher score for strikes near 30-delta (around 30% OTM)
                    optimal_distance = -15  # Approximately 30-delta for puts
                    distance_score = 1 - min(abs(distance_pct - optimal_distance) / 20, 1)
                    
                    # Calculate overall score based on ROI, distance, and market trend
                    score = ((annualized_roi / 100) * 0.6 + distance_score * 0.4) * bull_put_score_multiplier
                    
                    # Calculate probability of profit (approximation based on distance)
                    prob_of_profit = min(85, 50 + abs(distance_pct) * 1.5)
                    
                    # Add spread to opportunities
                    bull_put_opportunities.append({
                        'expiration': exp_date,
                        'days': days,
                        'type': 'Bull Put Spread',
                        'short_strike': short_put['strike'],
                        'long_strike': long_put['strike'],
                        'width': width,
                        'credit': credit,
                        'max_loss': max_loss,
                        'roi': roi,
                        'annualized_roi': annualized_roi,
                        'risk_reward': risk_reward_ratio,
                        'pop': prob_of_profit,
                        'short_distance': distance_pct,
                        'score': score
                    })
            
            # Find bear call spread opportunities
            otm_calls = calls[calls['strike'] > self.current_price].copy()
            if not otm_calls.empty:
                # Sort by strike ascending (closest to money first)
                otm_calls = otm_calls.sort_values('strike')
                
                # Take 10 closest strikes to current price for short call
                for short_idx, short_call in otm_calls.head(10).iterrows():
                    # Find potential long calls with higher strikes
                    long_calls = otm_calls[otm_calls['strike'] > short_call['strike']]
                    
                    if long_calls.empty:
                        continue
                        
                    # Get the next strike up for the long call
                    long_call = long_calls.iloc[0]
                    
                    # Calculate spread metrics
                    width = long_call['strike'] - short_call['strike']
                    
                    # Check if strikes are valid
                    if width < 1:
                        continue
                        
                    # Calculate potential credit
                    # If bid/ask are zero, use last price as a fallback
                    short_price = short_call['bid'] if short_call['bid'] > 0 else short_call['lastPrice'] * 0.9
                    long_price = long_call['ask'] if long_call['ask'] > 0 else long_call['lastPrice'] * 1.1
                    
                    credit = max(0.01, short_price - long_price)  # Ensure minimum credit
                    
                    if credit < min_credit:
                        continue
                        
                    max_loss = width - credit
                    risk_reward_ratio = max_loss / credit
                    
                    if risk_reward_ratio > max_risk_reward_ratio:
                        continue
                        
                    roi = (credit / max_loss) * 100
                    annualized_roi = roi * (365 / days)
                    
                    # Calculate distance from current price as percentage
                    distance_pct = ((short_call['strike'] - self.current_price) / self.current_price) * 100
                    
                    # Higher score for strikes near 30-delta (around 30% OTM)
                    optimal_distance = 15  # Approximately 30-delta for calls
                    distance_score = 1 - min(abs(distance_pct - optimal_distance) / 20, 1)
                    
                    # Calculate overall score based on ROI, distance, and market trend
                    score = ((annualized_roi / 100) * 0.6 + distance_score * 0.4) * bear_call_score_multiplier
                    
                    # Calculate probability of profit (approximation based on distance)
                    prob_of_profit = min(85, 50 + distance_pct * 1.5)
                    
                    # Add spread to opportunities
                    bear_call_opportunities.append({
                        'expiration': exp_date,
                        'days': days,
                        'type': 'Bear Call Spread',
                        'short_strike': short_call['strike'],
                        'long_strike': long_call['strike'],
                        'width': width,
                        'credit': credit,
                        'max_loss': max_loss,
                        'roi': roi,
                        'annualized_roi': annualized_roi,
                        'risk_reward': risk_reward_ratio,
                        'pop': prob_of_profit,
                        'short_distance': distance_pct,
                        'score': score
                    })
        
        # Sort opportunities by score
        bull_put_opportunities.sort(key=lambda x: x['score'], reverse=True)
        bear_call_opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Report top opportunities
        print("\n=== TOP BULL PUT SPREAD OPPORTUNITIES ===")
        if bull_put_opportunities:
            for i, opp in enumerate(bull_put_opportunities[:5]):
                print(f"\n{i+1}. {opp['type']} - Exp: {opp['expiration']} ({opp['days']} days)")
                print(f"   Short ${opp['short_strike']} Put / Long ${opp['long_strike']} Put (Width: ${opp['width']})")
                print(f"   Credit: ${opp['credit']:.2f}, Max Loss: ${opp['max_loss']:.2f}")
                print(f"   ROI: {opp['roi']:.2f}%, Annualized: {opp['annualized_roi']:.2f}%")
                print(f"   Risk/Reward: {opp['risk_reward']:.2f}, Est. Prob of Profit: {opp['pop']:.1f}%")
                print(f"   Short Strike Distance from Current Price: {opp['short_distance']:.2f}%")
                print(f"   Score: {opp['score']:.2f}")
        else:
            print("   No suitable bull put spread opportunities found")
            
        print("\n=== TOP BEAR CALL SPREAD OPPORTUNITIES ===")
        if bear_call_opportunities:
            for i, opp in enumerate(bear_call_opportunities[:5]):
                print(f"\n{i+1}. {opp['type']} - Exp: {opp['expiration']} ({opp['days']} days)")
                print(f"   Short ${opp['short_strike']} Call / Long ${opp['long_strike']} Call (Width: ${opp['width']})")
                print(f"   Credit: ${opp['credit']:.2f}, Max Loss: ${opp['max_loss']:.2f}")
                print(f"   ROI: {opp['roi']:.2f}%, Annualized: {opp['annualized_roi']:.2f}%")
                print(f"   Risk/Reward: {opp['risk_reward']:.2f}, Est. Prob of Profit: {opp['pop']:.1f}%")
                print(f"   Short Strike Distance from Current Price: {opp['short_distance']:.2f}%")
                print(f"   Score: {opp['score']:.2f}")
        else:
            print("   No suitable bear call spread opportunities found")
            
        return {
            'bull_put': bull_put_opportunities,
            'bear_call': bear_call_opportunities
        }

    def plot_price_history(self):
        """Plot price history with technical indicators"""
        if self.history is None or len(self.history) == 0:
            print("No historical data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Price and moving averages
        plt.subplot(2, 1, 1)
        plt.plot(self.history.index, self.history['Close'], label='Price')
        plt.plot(self.history.index, self.history['MA20'], label='20-day MA')
        plt.plot(self.history.index, self.history['MA50'], label='50-day MA')
        plt.plot(self.history.index, self.history['BB_upper'], 'r--', label='BB Upper')
        plt.plot(self.history.index, self.history['BB_lower'], 'r--', label='BB Lower')
        plt.fill_between(self.history.index, self.history['BB_upper'], self.history['BB_lower'], alpha=0.1, color='r')
        plt.title(f"{self.ticker_symbol} Price History")
        plt.legend()
        plt.grid(True)
        
        # RSI
        plt.subplot(2, 1, 2)
        plt.plot(self.history.index, self.history['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
        plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
        plt.title("RSI Indicator")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        plt.savefig(f"output/{self.ticker_symbol}_analysis.png")
        print(f"\nPrice history chart saved to output/{self.ticker_symbol}_analysis.png")
        plt.close()

def main():
    analyzer = TeslaCreditSpreadAnalyzer()
    
    # Load historical data and calculate technicals
    analyzer.load_history(period="6mo")
    
    # Plot price chart with indicators
    analyzer.plot_price_history()
    
    # Load option chains
    analyzer.load_option_chains(weeks=6)
    
    # Find credit spread opportunities with relaxed parameters
    opportunities = analyzer.find_credit_spread_opportunities(min_credit=0.10, max_risk_reward_ratio=10)

if __name__ == "__main__":
    main() 