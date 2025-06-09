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
    df.dropna(subset=['value', 'token', 'event_type', 'hash'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Custom Sorting and Renaming ---
    # Rule: For transactions sharing a hash where 'debit' and 'aave_supply' exist,
    # rename them to clarify the transaction flow.
    def has_debit_aave_supply_pair(s):
        return 'debit' in s.values and 'aave_supply' in s.values

    has_pair_series = df.groupby('hash')['event_type'].transform(has_debit_aave_supply_pair)
    
    debit_mask = (has_pair_series) & (df['event_type'] == 'debit')
    aave_mask = (has_pair_series) & (df['event_type'] == 'aave_supply')
    
    df.loc[debit_mask, 'event_type'] = 'deposit_into_aave'
    df.loc[aave_mask, 'event_type'] = 'receive_Atokens'

    # Rule: For transactions sharing a hash where 'aave_withdraw' and 'credit' exist,
    # rename them to clarify the transaction flow.
    def has_aave_withdraw_credit_pair(s):
        return 'aave_withdraw' in s.values and 'credit' in s.values

    has_withdraw_credit_pair_series = df.groupby('hash')['event_type'].transform(has_aave_withdraw_credit_pair)

    withdraw_mask = (has_withdraw_credit_pair_series) & (df['event_type'] == 'aave_withdraw')
    credit_mask = (has_withdraw_credit_pair_series) & (df['event_type'] == 'credit')

    df.loc[withdraw_mask, 'event_type'] = 'burn_Atokens'
    df.loc[credit_mask, 'event_type'] = 'receive_erc20_tokens'

    # Sort transactions to group by hash and apply specific ordering rules.
    # Hashes are ordered by their first transaction time. Within a hash group,
    # transactions are ordered alphabetically by event_type. This handles both
    # 'from_swap'/'to_swap' and the renamed Aave events.
    df['group_min_timestamp'] = df.groupby('hash')['timestamp'].transform('min')
    df.sort_values(by=['group_min_timestamp', 'hash', 'event_type'], inplace=True, ignore_index=True)
    df.drop(columns=['group_min_timestamp'], inplace=True)

    # --- Balance Calculation with Refined Double-Entry Logic ---
    balances = {}
    wallet_address = df['wallet'].iloc[0]

    for token in df['token'].unique():
        balances[token] = 0.0


    balances_history = []
    current_prices = {}

    for _, row in df.iterrows():
        token = row['token']
        event = row['event_type']
        value = row['value']
        from_address = row['from']
        tx_hash = row['hash']

        # Update prices from transaction data
        if row['value'] > 0 and row['value_usd'] > 0:
            current_prices[token] = row['value_usd'] / row['value']

        # Standard credits and debits
        if event in ['from_treasury_wallet', 'from_swap', 'credit', 'receive_erc20_tokens']:
            balances[token] += value
        elif event in ['to_treasury_wallet', 'to_swap', 'debit', 'deposit_into_aave']:
            balances[token] -= value

        # Aave Supply: Occurs when you send an asset to an Aave pool
        elif event == 'aave_supply' and from_address == wallet_address:
            balances[token] -= value 
        
        # Aave Mint: You receive an aToken after supplying the underlying asset
        elif event == 'aave_supply' and from_address != wallet_address:
             balances[token] += value
        
        elif event == 'receive_Atokens':
            balances[token] += value

        # Aave Withdraw: Occurs when you burn aTokens to get the underlying asset back
        elif event in ['aave_withdraw', 'burn_Atokens']:
            balances[token] -= value

        # --- NAV Calculation at each step ---
        nav = 0.0
        for t, b in balances.items():
            if abs(b) > 1e-9:
                price = current_prices.get(t, 1.0)
                if t.startswith('aArb') and t not in current_prices:
                    underlying = t.replace('aArb', '').replace('n', '')
                    if underlying in current_prices:
                        price = current_prices[underlying]
                nav += b * price

        # Log the state of balances after each transaction
        balance_snapshot = row.to_dict()
        balance_snapshot['nav'] = nav
        balance_snapshot['token_balances'] = balances.copy()
        balances_history.append(balance_snapshot)

    print("\n--- Final Token Balances ---")
    for token, balance in sorted(balances.items()):
        if abs(balance) < 1e-9:
            balance = 0.0
        print(f"{token}: {balance:,.6f}")

    # --- Balances History DataFrame ---
    balances_df = pd.DataFrame(balances_history)
    
    final_nav = balances_df['nav'].iloc[-1] if not balances_df.empty else 0.0
    print("-------------------------------------------------")
    print(f"Net Asset Value (NAV): ${final_nav:,.2f}")
    print("-------------------------------------------------")

    print("\n--- Balances History ---")
    print(balances_df[['timestamp', 'event_type', 'token', 'value', 'nav']])
    
    # --- Export to CSV ---
    # Define which columns to include in the final CSV file.
    # You can add, remove, or reorder columns from this list.
    columns_to_export = [
        'timestamp',
        'token',
        # 'wallet',
        # 'from',
        # 'token',
        'value',
        'value_usd',
        # 'transaction_type',
        # 'aave_event',
        'event_type',
        'nav',
        'from',
        'to',
        'hash',
        'token_balances',
        # 'wallet',
        # 'from',
        # 'token',
        'transaction_type',
        'aave_event',
        'wallet',
    ]
    balances_df[columns_to_export].to_csv('balances_history_v2.csv', index=False)


if __name__ == '__main__':
    # Make sure you have pandas installed: pip install pandas
    calculate_nav('raw_df_from_query.csv') 
