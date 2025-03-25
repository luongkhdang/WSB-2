import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath('src'))

from main import WSBTradingApp

def test_watchlist():
    # Initialize the trading app
    app = WSBTradingApp()
    
    # Get current symbols
    current_symbols = app.get_watchlist_symbols()
    print(f"Current watchlist has {len(current_symbols)} symbols")
    print(f"First 10 symbols: {current_symbols[:10]}")
    
    # Check if constant players are included
    constant_players_included = all(player in current_symbols for player in app.constant_players)
    print(f"All constant players included: {constant_players_included}")
    
    # Update the watchlist
    print("\nUpdating watchlist...")
    result = app.update_watchlist()
    print(f"Update result: {result}")
    
    # Check updated watchlist
    if result:
        updated_symbols = app.get_watchlist_symbols()
        print(f"\nUpdated watchlist has {len(updated_symbols)} symbols")
        print(f"First 10 symbols: {updated_symbols[:10]}")
        
        # Check if constant players are included after update
        updated_constant_players_included = all(player in updated_symbols for player in app.constant_players)
        print(f"All constant players included after update: {updated_constant_players_included}")
        
        # Print out the constant players
        print("\nConstant players that should be included:")
        for player in app.constant_players:
            included = player in updated_symbols
            print(f"- {player}: {'✓' if included else '✗'}")
        
    # Print the content of the watchlist file
    print("\nContent of the watchlist file:")
    with open(app.watchlist_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    test_watchlist() 