import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Connect to MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# Account info
print("\n===== ACCOUNT INFO =====\n")
account_info = mt5.account_info()
if account_info is not None:
    account_dict = account_info._asdict()
    for key, value in account_dict.items():
        print(f"{key}: {value}")

# Check closed positions (last 7 days)
print("\n===== CLOSED POSITIONS (LAST 7 DAYS) =====\n")
from_date = datetime.now() - timedelta(days=7)
to_date = datetime.now()

# Get history deals
deals = mt5.history_deals_get(from_date, to_date)
if deals is None:
    print("No deals found")
else:
    print(f"Found {len(deals)} deals")
    for deal in deals:
        deal_dict = deal._asdict()
        print(f"\nDeal ID: {deal_dict['ticket']}")
        print(f"Time: {datetime.fromtimestamp(deal_dict['time'])}")
        print(f"Symbol: {deal_dict['symbol']}")
        print(f"Type: {deal_dict['type']}")
        print(f"Entry: {deal_dict['entry']}")
        print(f"Volume: {deal_dict['volume']}")
        print(f"Price: {deal_dict['price']}")
        print(f"Profit: {deal_dict['profit']}")
        print(f"Comment: {deal_dict['comment']}")

# Check active positions
print("\n===== ACTIVE POSITIONS =====\n")
positions = mt5.positions_get()
if positions is None:
    print("No active positions")
else:
    print(f"Found {len(positions)} active positions")
    for position in positions:
        pos_dict = position._asdict()
        print(f"\nTicket: {pos_dict['ticket']}")
        print(f"Symbol: {pos_dict['symbol']}")
        print(f"Time: {datetime.fromtimestamp(pos_dict['time'])}")
        print(f"Type: {'Buy' if pos_dict['type'] == 0 else 'Sell'}")
        print(f"Volume: {pos_dict['volume']}")
        print(f"Open Price: {pos_dict['price_open']}")
        print(f"Current Price: {pos_dict['price_current']}")
        print(f"SL: {pos_dict['sl']}")
        print(f"TP: {pos_dict['tp']}")
        print(f"Profit: {pos_dict['profit']}")

# Shutdown
mt5.shutdown()
