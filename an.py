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
        {'aave_withdraw', 'credit'},
    ]
    # DECREASING_TX_TYPES = {'from_0x_settler_v1_9'}
    DECREASING_TX_TYPES = {'debit'}
    INCREASING_TX_TYPES = {'credit'}

    # Get indices of zero-impact transaction groups
    zero_impact_indices = set()
    grouped_by_hash = df_processed.groupby('hash')
    for _, group in grouped_by_hash:
        tx_types = set(group['transaction_type'])
        for group_def in ZERO_NAV_CHANGE_GROUPS:
            if group_def.issubset(tx_types):
                zero_impact_indices.update(group.index)
                break
    
    # Initialize nav_change column
    df_processed['nav_change'] = 0.0

    # Iterate through the DataFrame to calculate nav_change row by row
    for index, row in df_processed.iterrows():
        # Rule 1: Zero-impact groups
        if index in zero_impact_indices:
            # nav_change remains 0.0
            continue

        # Rule 2: Increasing tx types
        if row['transaction_type'] in INCREASING_TX_TYPES:
            df_processed.loc[index, 'nav_change'] = abs(row['value_usd'])
            continue

        # Rule 3: Decreasing tx types
        if row['transaction_type'] in DECREASING_TX_TYPES:
            df_processed.loc[index, 'nav_change'] = -abs(row['value_usd'])
            continue

        # Default rule for all other transactions
        is_incoming = row['to'] == wallet_address
        is_outgoing = row['from'] == wallet_address
        
        if is_incoming and not is_outgoing:
            df_processed.loc[index, 'nav_change'] = row['value_usd']
        elif is_outgoing and not is_incoming:
            df_processed.loc[index, 'nav_change'] = -row['value_usd']
        # For self-transfers or unhandled cases, nav_change remains 0
    
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
