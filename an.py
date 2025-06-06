import pandas as pd

def calculate_nav(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Net Asset Value (NAV) for a series of wallet transactions.

    Args:
        df: DataFrame with wallet transactions. 
            Expected columns: ['timestamp', 'wallet', 'network', 'token', 
                               'transaction_type', 'aave_event', 'value', 
                               'value_usd', 'hash', 'to', 'from']

    Returns:
        DataFrame with two new columns: 'nav_change' and 'nav'.
    """
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the wallet address we are analyzing
    wallet_address = df['wallet'].iloc[0]

    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Sort transactions chronologically.
    # Using mergesort for stable sort to preserve original order for ties.
    df_processed = df_processed.sort_values(by='timestamp', kind='mergesort').reset_index(drop=True)
    
    # --- NAV Calculation ---

    # Define rule sets
    ZERO_NAV_CHANGE_GROUPS = [
        {'to_0x_proxy_wallet', 'from_0x_proxy_wallet'},
        {'aave_supply', 'debit'},
    ]
    DECREASING_NAV_CHANGE_GROUPS = [{'debit'}]
    INCREASING_NAV_CHANGE_GROUPS = [{'credit'}]

    BUNDLED_TXN_RULES = [
        {
            'types': {'aave_withdraw', 'credit'},
            'operations': {
                'aave_withdraw': 'add',
                'credit': 'subtract'
            }
        }
    ]

    # Get indices for all rule groups, ensuring precedence
    zero_impact_indices = set()
    increasing_indices = set()
    decreasing_indices = set()
    bundled_indices = set()
    bundled_nav_changes = {}

    grouped_by_hash = df_processed.groupby('hash')
    for hash_val, group in grouped_by_hash:
        tx_types = set(group['transaction_type'])
        
        # Rule 0: Bundled Transactions (highest precedence)
        is_bundled = False
        for rule in BUNDLED_TXN_RULES:
            if rule['types'].issubset(tx_types):
                total_change = 0.0
                for index, row in group.iterrows():
                    op = rule['operations'].get(row['transaction_type'])
                    if op == 'add':
                        total_change += row['value_usd']
                    elif op == 'subtract':
                        total_change -= row['value_usd']
                
                bundled_nav_changes[hash_val] = total_change
                bundled_indices.update(group.index)
                is_bundled = True
                break
        if is_bundled:
            continue

        # Rule 1: Zero-impact groups
        is_zero_impact = False
        for group_def in ZERO_NAV_CHANGE_GROUPS:
            if group_def.issubset(tx_types):
                zero_impact_indices.update(group.index)
                is_zero_impact = True
                break
        if is_zero_impact:
            continue

        # Rule 2: Increasing groups
        is_increasing = False
        for group_def in INCREASING_NAV_CHANGE_GROUPS:
            if group_def.issubset(tx_types):
                increasing_indices.update(group.index)
                is_increasing = True
                break
        if is_increasing:
            continue

        # Rule 3: Decreasing groups
        for group_def in DECREASING_NAV_CHANGE_GROUPS:
            if group_def.issubset(tx_types):
                decreasing_indices.update(group.index)
                break
    
    # Initialize nav_change column
    df_processed['nav_change'] = 0.0

    # Apply bundled changes first
    for hash_val, change in bundled_nav_changes.items():
        # Get the first index of the group to assign the net change
        group_indices = df_processed[df_processed['hash'] == hash_val].index
        if len(group_indices) > 0:
            first_index = group_indices[0]
            df_processed.loc[first_index, 'nav_change'] = change
            # Other indices in the group will have nav_change = 0 by default

    # Iterate through the DataFrame to calculate nav_change for other rules
    for index, row in df_processed.iterrows():
        # Skip if already handled by bundled rule
        if index in bundled_indices:
            continue

        # Rule 1: Zero-impact groups
        if index in zero_impact_indices:
            df_processed.loc[index, 'nav_change'] = 0.0
        
        # Rule 2: Increasing groups
        elif index in increasing_indices:
            df_processed.loc[index, 'nav_change'] = abs(row['value_usd'])

        # Rule 3: Decreasing groups
        elif index in decreasing_indices:
            df_processed.loc[index, 'nav_change'] = -abs(row['value_usd'])

        # Default rule for all other transactions
        else:
            is_incoming = row['to'] == wallet_address
            is_outgoing = row['from'] == wallet_address
            
            if is_incoming and not is_outgoing:
                df_processed.loc[index, 'nav_change'] = row['value_usd']
            elif is_outgoing and not is_incoming:
                df_processed.loc[index, 'nav_change'] = -row['value_usd']
    
    # Calculate cumulative NAV
    df_processed['nav'] = df_processed['nav_change'].cumsum()
    
    return df_processed

if __name__ == '__main__':
    try:
        raw_df = pd.read_csv('raw_df_from_query.csv')
        
        nav_df = calculate_nav(raw_df)

        pd.set_option('display.max_rows', 528)
        print("NAV calculation results (first 100 rows):")
        print(nav_df[['timestamp', 'token', 'transaction_type', 'value_usd', 'nav_change', 'nav']].head(528))

        # To save the results to a new CSV file:
        nav_df.to_csv('nav_results.csv', index=False)
        print("\nResults saved to nav_results.csv")

    except FileNotFoundError:
        print("Error: 'raw_df_from_query.csv' not found. Make sure the file is in the same directory.") 
