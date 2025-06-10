import pandas as pd

def calculate_nav(csv_path: str):
    """
    Calculates the Net Asset Value (NAV) from a transaction history CSV.

    Args:
        csv_path:str
    """
    df = pd.read_csv(csv_path)

    df['event_type'] = df['transaction_type']
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Helper functions to locate txns that have the same hash by event
    def has_debit_aave_supply_pair(s):
        return 'debit' in s.values and 'aave_supply' in s.values
    def has_aave_withdraw_credit_pair(s):
        return 'aave_withdraw' in s.values and 'credit' in s.values
    def has_swap_pair(s):
        return 'from_swap' in s.values and 'to_swap' in s.values

    # Locate Aave supply bundles by event
    has_pair_series = df.groupby('hash')['event_type'].transform(has_debit_aave_supply_pair)
    debit_mask = (has_pair_series) & (df['event_type'] == 'debit')
    aave_mask = (has_pair_series) & (df['event_type'] == 'aave_supply')
    df.loc[debit_mask, 'event_type'] = 'deposit_into_aave'
    df.loc[aave_mask, 'event_type'] = 'receive_Atokens'

    # Locate Aave withdrawal bundles by event
    has_withdraw_credit_pair_series = df.groupby('hash')['event_type'].transform(has_aave_withdraw_credit_pair)
    withdraw_mask = (has_withdraw_credit_pair_series) & (df['event_type'] == 'aave_withdraw')
    credit_mask = (has_withdraw_credit_pair_series) & (df['event_type'] == 'credit')
    df.loc[withdraw_mask, 'event_type'] = 'burn_Atokens'
    df.loc[credit_mask, 'event_type'] = 'receive_erc20_tokens'

    # Locate Swap bundles by event
    has_swap_pair_series = df.groupby('hash')['event_type'].transform(has_swap_pair)
    from_swap_mask = (has_swap_pair_series) & (df['event_type'] == 'from_swap')
    to_swap_mask = (has_swap_pair_series) & (df['event_type'] == 'to_swap')
    df.loc[from_swap_mask, 'event_type'] = 'send_for_swap'
    df.loc[to_swap_mask, 'event_type'] = 'receive_from_swap'

    # Order bundles by event
    df['group_min_timestamp'] = df.groupby('hash')['timestamp'].transform('min')
    df.sort_values(by=['group_min_timestamp', 'hash', 'event_type', 'timestamp'], inplace=True, ignore_index=True)
    df.drop(columns=['group_min_timestamp'], inplace=True)

    balances = {}

    for token in df['token'].unique():
        balances[token] = 0.0

    nav_values = []
    token_balances_history = []
    current_prices = {}
    nav = 0.0  

    for index, row in df.iterrows():
        token = row['token']
        event = row['event_type']
        value = row['value']
        value_usd = pd.to_numeric(row['value_usd'], errors='coerce') if 'value_usd' in row else 0.0
        tx_hash = row['hash']

        if event == 'from_treasury_wallet':
            balances[token] += value
        elif event == 'to_treasury_wallet':
            balances[token] -= value

        elif event == 'aave_supply':
            balances[token] += value
        elif event == 'deposit_into_aave':
            balances[token] = 0
        elif event == 'receive_Atokens':
            balances[token] += value

        elif event == 'aave_withdraw':
            balances[token] = 0
        elif event == 'burn_Atokens':
            balances[token] = 0
        elif event == 'receive_erc20_tokens':
            pass

        elif event == 'send_for_swap':
            # Example hash 0x06b17ac615b7fe83c8b42a3ec4e4db58a0a7514e9219f7047acd9e5a375fedff
            pass
        elif event == 'receive_from_swap':
            pass
        elif event == 'to_swap':
            balances[token] = 0
        elif event == 'from_swap':
            balances[token] -= value

        if row['value'] > 0 and row['value_usd'] > 0:
            current_prices[token] = row['value_usd'] / row['value']

        current_balance = balances[token]

        if current_balance <= 0.0:
            nav_value = nav_values[-1]
        else:
            nav_value = current_balance

        nav_values.append(nav_value)
        token_balances_history.append(balances.copy())

    print("\nToken balance sanity check:")
    for token, balance in sorted(balances.items()):
        print(f"{token}: {balance}")

    df['nav'] = nav_values
    df['token_balances'] = token_balances_history

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
        'aave_event',
        'wallet',
    ]
    df[columns_to_export].to_csv('balances_history_v2.csv', index=False)


if __name__ == '__main__':
    calculate_nav('raw_df_from_query.csv') 
