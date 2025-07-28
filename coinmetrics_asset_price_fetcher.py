import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetPriceRequest:
    """Request model for asset price fetching with validation."""
    
    def __init__(self, assets: List[str], start_date: str = "2024-12-31", 
                 frequency: str = "1d", page_size: int = 900):
        self.assets = self._validate_assets(assets)
        self.start_date = self._validate_start_date(start_date)
        self.frequency = frequency
        self.page_size = page_size
    
    def _validate_assets(self, assets: List[str]) -> List[str]:
        """Validate and normalize asset list."""
        if not assets:
            raise ValueError("Assets list cannot be empty")
        return [asset.lower().strip() for asset in assets]
    
    def _validate_start_date(self, start_date: str) -> str:
        """Validate start date format."""
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            return start_date
        except ValueError:
            raise ValueError("start_date must be in YYYY-MM-DD format")


class CoinMetricsAssetPriceFetcher:
    """
    A Pythonic class to fetch asset prices from CoinMetrics API.
    Returns transposed DataFrames with columns for each asset.
    """
    
    def __init__(self, api_key: str, timeout: int = 30):
        """
        Initialize the asset price fetcher.
        
        Args:
            api_key: CoinMetrics API key
            timeout: Request timeout in seconds
            
        Raises:
            ValueError: If API key is not provided
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.base_url = "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
        self.timeout = timeout
        self.session = requests.Session()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.session.close()
    
    def _build_params(self, request: AssetPriceRequest) -> Dict[str, str]:
        """Build API request parameters."""
        return {
            "assets": ",".join(request.assets),
            "metrics": "PriceUSD",
            "frequency": request.frequency,
            "start_time": request.start_date,
            "pretty": "true",
            "paging_from": "start",
            "page_size": str(request.page_size),
            "api_key": self.api_key,
        }
    
    def _fetch_page(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetch a single page from the API.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If API request fails
        """
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
    
    def _fetch_all_pages(self, request: AssetPriceRequest) -> List[Dict[str, Any]]:
        """
        Fetch all pages of data with pagination handling.
        
        Args:
            request: Validated request parameters
            
        Returns:
            List of all data records from all pages
            
        Raises:
            requests.RequestException: If any API request fails
        """
        all_data = []
        next_page_url = self.base_url
        params = self._build_params(request)
        is_first_page = True
        page_count = 0
        
        while next_page_url:
            try:
                page_count += 1
                logger.info(f"Fetching page {page_count}...")
                
                response_data = self._fetch_page(
                    next_page_url, 
                    params if is_first_page else None
                )
                
                data = response_data.get("data", [])
                if not data:
                    logger.warning(f"No data received from API page {page_count}")
                    break
                
                all_data.extend(data)
                next_page_url = response_data.get("next_page_url")
                is_first_page = False
                
                logger.info(f"Page {page_count}: {len(data)} records. Total: {len(all_data)}")
                
            except Exception as e:
                logger.error(f"Error fetching page {page_count}: {e}")
                raise
        
        logger.info(f"Completed fetching {page_count} pages with {len(all_data)} total records")
        return all_data
    
    def _process_data_to_dataframe(self, raw_data: List[Dict[str, Any]], assets: List[str]) -> pd.DataFrame:
        """
        Process raw API data into a transposed DataFrame with asset columns.
        
        Args:
            raw_data: Raw data from API
            assets: List of requested assets
            
        Returns:
            Transposed DataFrame with time and asset price columns
            
        Raises:
            ValueError: If data processing fails
        """
        if not raw_data:
            raise ValueError("No data to process")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Ensure required columns exist
        required_columns = ['time', 'asset', 'PriceUSD']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Data cleaning and type conversion
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['PriceUSD'] = pd.to_numeric(df['PriceUSD'], errors='coerce')
        df['asset'] = df['asset'].str.lower()
        
        # Remove rows with invalid data
        initial_rows = len(df)
        df = df.dropna(subset=['time', 'PriceUSD'])
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows with invalid data")
        
        # Pivot to get assets as columns (transposed format)
        try:
            pivoted_df = df.pivot(index='time', columns='asset', values='PriceUSD')
        except ValueError as e:
            logger.error(f"Error pivoting data: {e}")
            raise ValueError(f"Failed to transpose data: {e}")
        
        # Ensure all requested assets are present as columns
        missing_assets = []
        for asset in assets:
            if asset not in pivoted_df.columns:
                missing_assets.append(asset)
                pivoted_df[asset] = None
        
        if missing_assets:
            logger.warning(f"Assets not found in API response: {missing_assets}")
        
        # Reorder columns to match requested asset order
        available_assets = [asset for asset in assets if asset in pivoted_df.columns]
        pivoted_df = pivoted_df[available_assets]
        
        # Rename columns to add _price_usd suffix
        column_mapping = {asset: f"{asset}_price_usd" for asset in available_assets}
        pivoted_df = pivoted_df.rename(columns=column_mapping)
        
        # Reset index to make time a regular column
        pivoted_df = pivoted_df.reset_index()
        
        # Sort by time for consistency
        pivoted_df = pivoted_df.sort_values('time').reset_index(drop=True)
        
        return pivoted_df
    
    def get_asset_prices(
        self, 
        assets: List[str], 
        start_date: str = "2024-12-31",
        frequency: str = "1d",
        page_size: int = 900
    ) -> pd.DataFrame:
        """
        Fetch asset prices and return a transposed DataFrame.
        
        Args:
            assets: List of asset symbols (e.g., ['usdc', 'usdt', 'dai'])
            start_date: Start date in YYYY-MM-DD format
            frequency: Data frequency ('1d', '1h', etc.)
            page_size: Number of records per API page
            
        Returns:
            DataFrame with 'time' column and one column per asset containing USD prices
            
        Raises:
            ValueError: If input validation fails or no data is received
            requests.RequestException: If API request fails
        """
        # Validate input parameters
        try:
            request = AssetPriceRequest(
                assets=assets,
                start_date=start_date,
                frequency=frequency,
                page_size=page_size
            )
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        logger.info(f"Fetching {frequency} prices for assets: {request.assets} from {start_date}")
        
        # Fetch and process data
        try:
            raw_data = self._fetch_all_pages(request)
            if not raw_data:
                raise ValueError("No data received from API")
            
            # Process into transposed DataFrame
            df = self._process_data_to_dataframe(raw_data, request.assets)
            
            logger.info(f"Successfully processed {len(df)} price records for {len(request.assets)} assets")
            logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch asset prices: {e}")
            raise
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "asset_prices.parquet") -> None:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Raises:
            Exception: If file saving fails
        """
        try:
            df.to_parquet(filename, compression="snappy", index=False)
            logger.info(f"Successfully saved {len(df)} records to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {e}")
            raise
    
    def get_asset_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate summary statistics for each asset.
        
        Args:
            df: DataFrame with asset price data
            
        Returns:
            Dictionary with summary stats for each asset (keyed by asset name without suffix)
        """
        summary = {}
        asset_columns = [col for col in df.columns if col != 'time' and col.endswith('_price_usd')]
        
        for asset_column in asset_columns:
            # Extract asset name by removing the '_price_usd' suffix
            asset_name = asset_column.replace('_price_usd', '')
            prices = df[asset_column].dropna()
            
            if len(prices) > 0:
                summary[asset_name] = {
                    'count': len(prices),
                    'mean_price': float(prices.mean()),
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'latest_price': float(prices.iloc[-1]) if len(prices) > 0 else None,
                    'price_change_pct': float(((prices.iloc[-1] / prices.iloc[0]) - 1) * 100) if len(prices) > 1 else 0
                }
            else:
                summary[asset_name] = {'count': 0, 'error': 'No valid price data'}
        
        return summary
    
    def merge_with_asset_amounts(
        self, 
        price_df: pd.DataFrame, 
        amounts_csv_path: str,
        timestamp_col: str = "BLOCK_TIMESTAMP"
    ) -> pd.DataFrame:
        """
        Merge price DataFrame with asset amounts CSV and calculate total USD values.
        
        Args:
            price_df: DataFrame with asset prices (from get_asset_prices)
            amounts_csv_path: Path to CSV file with asset amounts
            timestamp_col: Name of timestamp column in amounts CSV
            
        Returns:
            Merged DataFrame with total_price_usd column
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If merge fails or required columns are missing
        """
        try:
            # Load amounts CSV
            amounts_df = pd.read_csv(amounts_csv_path)
            logger.info(f"Loaded amounts CSV with shape: {amounts_df.shape}")
            logger.info(f"Amounts CSV columns: {list(amounts_df.columns)}")
            
        except FileNotFoundError:
            logger.error(f"Amounts CSV file not found: {amounts_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading amounts CSV: {e}")
            raise
        
        # Validate required columns exist
        if timestamp_col not in amounts_df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in amounts CSV")
        
        # Create asset mapping from CSV columns to price columns
        asset_mapping = self._create_asset_mapping(amounts_df.columns, price_df.columns)
        logger.info(f"Asset mapping: {asset_mapping}")
        
        # Prepare DataFrames for merging
        try:
            # Convert timestamps to dates for merging
            price_df_copy = price_df.copy()
            amounts_df_copy = amounts_df.copy()
            
            # Handle different timestamp formats
            price_df_copy['merge_date'] = pd.to_datetime(price_df_copy['time']).dt.date
            amounts_df_copy['merge_date'] = pd.to_datetime(amounts_df_copy[timestamp_col]).dt.date
            
            logger.info(f"Price data date range: {price_df_copy['merge_date'].min()} to {price_df_copy['merge_date'].max()}")
            logger.info(f"Amounts data date range: {amounts_df_copy['merge_date'].min()} to {amounts_df_copy['merge_date'].max()}")
            
        except Exception as e:
            logger.error(f"Error processing timestamps: {e}")
            raise ValueError(f"Failed to process timestamps: {e}")
        
        # Perform merge on dates
        try:
            merged_df = pd.merge(
                amounts_df_copy, 
                price_df_copy, 
                on='merge_date', 
                how='inner'
            )
            logger.info(f"Merged DataFrame shape: {merged_df.shape}")
            
            if merged_df.empty:
                logger.warning("No matching dates found between price and amounts data")
                return merged_df
            
        except Exception as e:
            logger.error(f"Error during merge: {e}")
            raise ValueError(f"Failed to merge DataFrames: {e}")
        
        # Calculate total USD value
        merged_df = self._calculate_total_usd_value(merged_df, asset_mapping)
        
        # Clean up temporary columns and reorder
        columns_to_keep = [timestamp_col, 'time', 'merge_date'] + \
                         [col for col in merged_df.columns if col.endswith('_price_usd')] + \
                         [col for col in amounts_df.columns if col != timestamp_col] + \
                         ['total_price_usd']
        
        # Only keep columns that actually exist
        final_columns = [col for col in columns_to_keep if col in merged_df.columns]
        merged_df = merged_df[final_columns]
        
        # Sort by timestamp
        merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info(f"Final merged DataFrame shape: {merged_df.shape}")
        logger.info(f"Final columns: {list(merged_df.columns)}")
        
        return merged_df
    
    def _create_asset_mapping(self, amounts_columns: List[str], price_columns: List[str]) -> Dict[str, str]:
        """
        Create mapping from amount column names to price column names.
        
        Args:
            amounts_columns: Column names from amounts CSV
            price_columns: Column names from price DataFrame
            
        Returns:
            Dictionary mapping amount columns to price columns
        """
        asset_mapping = {}
        
        # Define base asset mappings (handles both direct and aToken variants)
        base_mappings = {
            'DAI': 'dai_price_usd',
            'USDC': 'usdc_price_usd', 
            'USDT': 'usdt_price_usd',
            'aEthDai': 'dai_price_usd',
            'aEthUSDC': 'usdc_price_usd',
            'aEthUSDT': 'usdt_price_usd'
        }
        
        # Create mapping for columns that exist in both datasets
        for amount_col in amounts_columns:
            if amount_col in base_mappings:
                price_col = base_mappings[amount_col]
                if price_col in price_columns:
                    asset_mapping[amount_col] = price_col
                else:
                    logger.warning(f"Price column '{price_col}' not found for asset '{amount_col}'")
        
        return asset_mapping
    
    def _calculate_total_usd_value(self, df: pd.DataFrame, asset_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Calculate total USD value by multiplying amounts by prices.
        
        Args:
            df: Merged DataFrame
            asset_mapping: Mapping from amount columns to price columns
            
        Returns:
            DataFrame with total_price_usd column added
        """
        df = df.copy()
        df['total_price_usd'] = 0.0
        
        for amount_col, price_col in asset_mapping.items():
            if amount_col in df.columns and price_col in df.columns:
                # Fill NaN values with 0 for calculation
                amounts = df[amount_col].fillna(0)
                prices = df[price_col].fillna(0)
                
                # Calculate USD value for this asset
                usd_value = amounts * prices
                df['total_price_usd'] += usd_value
                
                logger.debug(f"Added {amount_col} * {price_col} to total")
            else:
                logger.warning(f"Missing columns for calculation: {amount_col} or {price_col}")
        
        # Round to reasonable precision
        df['total_price_usd'] = df['total_price_usd'].round(6)
        
        return df


# Example usage and testing
def main():
    """Example usage of the CoinMetricsAssetPriceFetcher class."""
    api_key = ""
    assets = ["usdc", "usdt", "dai"]
    
    try:
        with CoinMetricsAssetPriceFetcher(api_key) as fetcher:
            # Example 1: Basic price fetching
            print("=== Example 1: Basic Asset Price Fetching ===")
            df = fetcher.get_asset_prices(
                assets=assets,
                start_date="2024-12-31"
            )
            
            print(f"Retrieved data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst 3 rows:")
            print(df.head(3))
            
            # Generate summary statistics
            summary = fetcher.get_asset_summary(df)
            print(f"\nAsset Summary:")
            for asset, stats in summary.items():
                if 'error' not in stats:
                    print(f"{asset.upper()}: Latest=${stats['latest_price']:.4f}, Change={stats['price_change_pct']:.2f}%")
            
            # Save to file
            fetcher.save_to_parquet(df, "stablecoin_prices.parquet")
            
            # Example 2: Merge with asset amounts (if CSV exists)
            amounts_csv = "example_assets_stables.csv"
            try:
                print(f"\n=== Example 2: Merge with Asset Amounts ===")
                merged_df = fetcher.merge_with_asset_amounts(
                    price_df=df,
                    amounts_csv_path=amounts_csv
                )
                merged_df.to_csv("merged_df_output.csv")
                if not merged_df.empty:
                    print(f"Merged data shape: {merged_df.shape}")
                    print(f"Merged columns: {list(merged_df.columns)}")
                    print("\nMerged data with total USD values:")
                    print(merged_df.head())
                    
                    # Show total USD values
                    if 'total_price_usd' in merged_df.columns:
                        total_sum = merged_df['total_price_usd'].sum()
                        print(f"\nTotal portfolio value: ${total_sum:.2f}")
                else:
                    print("No matching dates found for merge.")
                    
            except FileNotFoundError:
                print(f"Amounts CSV '{amounts_csv}' not found. Skipping merge example.")
            except Exception as merge_error:
                print(f"Merge example failed: {merge_error}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
