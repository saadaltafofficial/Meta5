import MetaTrader5 as mt5
import time
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize MT5
if not mt5.initialize():
    logger.error(f"MT5 initialization failed: {mt5.last_error()}")
    exit()

# Check connection
account_info = mt5.account_info()
if account_info:
    account_dict = account_info._asdict()
    print(f"Connected to MT5 account: {account_dict['login']}")
    print(f"Balance: ${account_dict['balance']:.2f}")
    print(f"Leverage: 1:{account_dict['leverage']}")
    
    # Get EURUSD info
    symbol = "EURUSD"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        mt5.shutdown()
        exit()
    
    # Enable symbol for trading
    if not symbol_info.visible:
        print(f"Symbol {symbol} is not visible, enabling...")
        mt5.symbol_select(symbol, True)
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get {symbol} tick")
        mt5.shutdown()
        exit()
    
    ask = tick.ask
    bid = tick.bid
    print(f"{symbol} - Bid: {bid}, Ask: {ask}")
    
    # Execute a BUY trade
    lot_size = 0.01  # Micro lot
    point = mt5.symbol_info(symbol).point
    price = ask
    sl = bid - 100 * point  # 10 pip stop loss
    tp = bid + 200 * point  # 20 pip take profit
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "MCP Trader Test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    
    print("Sending trade request...")
    result = mt5.order_send(request)
    print(f"Result code: {result.retcode}")
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Trade executed successfully!")
        print(f"Order ID: {result.order}")
        print(f"Execution price: {result.price}")
        
        # Check positions
        time.sleep(1)
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                pos_dict = position._asdict()
                print(f"Position: {pos_dict['ticket']}")
                print(f"  Type: {'BUY' if pos_dict['type'] == mt5.POSITION_TYPE_BUY else 'SELL'}")
                print(f"  Volume: {pos_dict['volume']}")
                print(f"  Open price: {pos_dict['price_open']}")
                print(f"  Current price: {pos_dict['price_current']}")
                print(f"  Profit: ${pos_dict['profit']:.2f}")
    else:
        print(f"Trade execution failed with error code: {result.retcode}")
        print(f"Comment: {result.comment}")

else:
    print("Not connected to MT5 account")

# Shutdown MT5
mt5.shutdown()
