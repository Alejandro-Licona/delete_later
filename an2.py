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

    # Pre-calculate event sets for each hash to make loop logic cleaner
    hash_events = df.groupby('hash')['event_type'].apply(set).to_dict()

    balances_history = []
    current_prices = {}

    for index, row in df.iterrows():
        token = row['token']
        event = row['event_type']
        value = row['value']
        from_address = row['from']
        tx_hash = row['hash']

        # Update prices from transaction data
        if row['value'] > 0 and row['value_usd'] > 0:
            current_prices[token] = row['value_usd'] / row['value']

        # --- Balance Calculation Logic ---
        # This section applies the new balance rules based on event type.
        # It is structured to handle paired transactions first, then fall back
        # to standard debit/credit logic for individual transactions.
        events_in_hash = hash_events.get(tx_hash, set())

        # Group 1 & 2: Paired Aave Transactions
        if event == 'deposit_into_aave' and 'receive_Atokens' in events_in_hash:
            pass
        elif event == 'receive_Atokens' and 'deposit_into_aave' in events_in_hash:
            if index > 0 and df.loc[index - 1, 'hash'] == tx_hash:
                deposit_info = df.loc[index - 1]
                balances[deposit_info['token']] -= deposit_info['value']
            balances[token] += value 
        
        elif event == 'burn_Atokens' and 'receive_erc20_tokens' in events_in_hash:
            pass
        elif event == 'receive_erc20_tokens' and 'burn_Atokens' in events_in_hash:
            if index > 0 and df.loc[index - 1, 'hash'] == tx_hash:
                burn_info = df.loc[index - 1]
                balances[burn_info['token']] -= burn_info['value']
            balances[token] += value

        # Group 3: Paired Swap Transactions
        elif event == 'from_swap' and 'to_swap' in events_in_hash:
            pass
        elif event == 'to_swap' and 'from_swap' in events_in_hash:
            if index > 0 and df.loc[index - 1, 'hash'] == tx_hash:
                from_info = df.loc[index - 1]
                balances[from_info['token']] -= from_info['value']
            balances[token] += value  # Rule 6: Balance of new token increases

        # Default Debit/Credit logic for all other transactions
        else:
            # Standard credits (assets coming into the wallet)
            if event in ['from_treasury_wallet']:
                balances[token] += value
            # Standard debits (assets leaving the wallet)
            elif event in ['to_treasury_wallet']:
                balances[token] -= value
            elif event == 'to_swap':
                pass
            elif event == 'from_swap':
                # balances[token] += value
                pass
            # Unpaired Aave Supply (asset leaves wallet)
            elif event == 'aave_supply':
                pass
            # Unpaired Aave Withdraw (aToken burned)
            elif event == 'aave_withdraw':
                pass

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

    print("\n--- Balances History ---")
    print(balances_df[['timestamp', 'event_type', 'token', 'value', 'nav']])
    
    # --- Export to CSV ---
    # Define which columns to include in the final CSV file.
    # You can add, remove, or reorder columns from this list.
    columns_to_export = [
        'timestamp',
        'token',
        'value',
        'value_usd',
        'event_type',
        'nav',
        'hash',
        'from',
        'to',
        'token_balances',
        # 'transaction_type',
        'aave_event',
        'wallet',
    ]
    balances_df[columns_to_export].to_csv('balances_history_v2.csv', index=False)


if __name__ == '__main__':
    # Make sure you have pandas installed: pip install pandas
    calculate_nav('raw_df_from_query.csv') 
