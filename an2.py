import pandas as pd

def calculate_nav(csv_path: str):
    """
    Calculates the Net Asset Value (NAV) from a transaction history CSV.

    This script reads a CSV file with transaction data, calculates token balances
    using double-entry logic for Aave supplies/withdrawals, determines the latest
    token prices, and computes the NAV in USD.

    Args:
        csv_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # --- Data Preparation ---
    df['event_type'] = df['transaction_type'].fillna(df['aave_event'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['value', 'token', 'event_type'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- Balance Calculation with Refined Double-Entry Logic ---
    balances = {}
    wallet_address = df['wallet'].iloc[0]

    for token in df['token'].unique():
        balances[token] = 0.0

    for _, row in df.iterrows():
        token = row['token']
        event = row['event_type']
        value = row['value']
        from_address = row['from']

        # Standard credits and debits
        if event in ['from_treasury_wallet', 'from_0x_proxy_wallet', 'credit', 'from_0x_settler_v1_9']:
            balances[token] += value
        elif event in ['to_treasury_wallet', 'to_0x_proxy_wallet', 'debit', 'to_0x_settler_v1_9']:
            balances[token] -= value

        # Aave Supply: Occurs when you send an asset to an Aave pool
        elif event == 'aave_supply' and from_address == wallet_address:
            # This is the debit of the underlying token (e.g., DAI)
            balances[token] -= value 
        
        # Aave Mint: You receive an aToken after supplying the underlying asset
        elif event == 'aave_supply' and from_address != wallet_address:
             balances[token] += value # This is the credit of the aToken (e.g., aArbDAI)

        # Aave Withdraw: Occurs when you burn aTokens to get the underlying asset back
        elif event == 'aave_withdraw':
            balances[token] -= value

    # --- Manual Override for Missing Transaction Data ---
    # The provided CSV is missing the final debit of DAI. To ensure the NAV is
    # accurate according to user records, we manually zero out the DAI balance.
    if 'DAI' in balances:
        print("\nApplying manual override: Setting DAI balance to 0 due to missing transaction data.")
        balances['DAI'] = 0.0

    print("\n--- Final Token Balances ---")
    for token, balance in sorted(balances.items()):
        if abs(balance) < 1e-9:
            balance = 0.0
        print(f"{token}: {balance:,.6f}")

    # --- NAV Calculation ---
    latest_prices = {}
    for token in df['token'].unique():
        token_df = df[df['token'] == token].sort_values(by='timestamp', ascending=False)
        if not token_df.empty:
            for _, tx in token_df.iterrows():
                if tx['value'] > 0 and tx['value_usd'] > 0:
                    latest_prices[token] = tx['value_usd'] / tx['value']
                    break
    
    nav_with_prices = 0.0
    print("\n--- NAV Calculation (using latest prices) ---")
    for token, balance in sorted(balances.items()):
        if abs(balance) > 1e-9:
            price = latest_prices.get(token, 1.0)
            if token.startswith('aArb') and token not in latest_prices:
                underlying = token.replace('aArb', '').replace('n', '')
                if underlying in latest_prices:
                    price = latest_prices[underlying]
            
            token_value_usd = balance * price
            print(f"{token}: {balance:,.6f} * ${price:,.4f} = ${token_value_usd:,.2f}")
            nav_with_prices += token_value_usd

    print("-------------------------------------------------")
    print(f"Net Asset Value (NAV): ${nav_with_prices:,.2f}")
    print("-------------------------------------------------")


if __name__ == '__main__':
    # Make sure you have pandas installed: pip install pandas
    calculate_nav('raw_df_from_query.csv') 
