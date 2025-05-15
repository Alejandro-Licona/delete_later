import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import time


class KaikoAPI:
    """
    A class to interact with the Kaiko API for cryptocurrency market data.
    Provides methods to fetch market data and DeFi protocol events.
    """
    
    BASE_URL = "https://us.market-api.kaiko.io/v2/data"
    EU_BASE_URL = "https://eu.market-api.kaiko.io/v2/data"
    
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize the KaikoAPI client.
        
        Args:
            api_key: Your Kaiko API key
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {'Accept': 'application/json', 'X-Api-Key': api_key}
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict:
        """
        Make a request to the Kaiko API with retry logic.
        
        Args:
            url: The API endpoint URL
            params: Query parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails after retries
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                attempts += 1
                if attempts >= self.max_retries:
                    raise e
                # Apply exponential backoff
                time.sleep(self.retry_delay * (2 ** (attempts - 1)))
    
    def get_ohlcv(self, 
                 exchange: str, 
                 instrument_class: str, 
                 pair: str, 
                 interval: str = "1d", 
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 sort: str = "desc",
                 page_size: int = 100) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a specific trading pair.
        
        Args:
            exchange: Exchange code (e.g., 'cbse' for Coinbase)
            instrument_class: Instrument class (e.g., 'spot')
            pair: Trading pair (e.g., 'btc-usd')
            interval: Time interval (e.g., '1d', '1h', '1m')
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            sort: Sort order ('asc' or 'desc')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the OHLCV data
        """
        url = f"{self.BASE_URL}/trades.v1/exchanges/{exchange}/{instrument_class}/{pair}/aggregations/ohlcv"
        params = {
            "interval": interval,
            "sort": sort,
            "page_size": page_size
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
            
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
            
    def get_vwap(self, 
                exchange: str, 
                instrument_class: str, 
                pair: str, 
                interval: str = "1d", 
                start_time: Optional[str] = None,
                end_time: Optional[str] = None,
                sort: str = "desc",
                page_size: int = 100) -> pd.DataFrame:
        """
        Get VWAP (Volume-Weighted Average Price) data for a specific trading pair.
        
        Args:
            exchange: Exchange code (e.g., 'cbse' for Coinbase)
            instrument_class: Instrument class (e.g., 'spot')
            pair: Trading pair (e.g., 'btc-usd')
            interval: Time interval (e.g., '1d', '1h', '1m')
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            sort: Sort order ('asc' or 'desc')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the VWAP data
        """
        url = f"{self.BASE_URL}/trades.v1/exchanges/{exchange}/{instrument_class}/{pair}/aggregations/vwap"
        params = {
            "interval": interval,
            "sort": sort,
            "page_size": page_size
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
            
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
    
    def get_asset_metrics(self,
                         asset: str,
                         start_time: str,
                         end_time: str,
                         interval: str = "1d",
                         sources: str = "false",
                         page_size: int = 100) -> pd.DataFrame:
        """
        Get asset metrics data for a specific cryptocurrency.
        
        Args:
            asset: Asset code (e.g., 'btc', 'eth', 'agix')
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            interval: Time interval (e.g., '1d', '1h', '1m')
            sources: Whether to include sources ('true' or 'false')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the asset metrics data
        """
        url = f"{self.BASE_URL}/analytics.v2/asset_metrics"
        params = {
            "asset": asset,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval,
            "sources": sources,
            "page_size": page_size
        }
        
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
    
    def get_exchange_metrics(self,
                           exchange: str,
                           start_time: str,
                           end_time: str,
                           interval: str = "1d",
                           sources: str = "false",
                           page_size: int = 100) -> pd.DataFrame:
        """
        Get exchange metrics data for a specific cryptocurrency exchange.
        
        Args:
            exchange: Exchange code (e.g., 'cbse' for Coinbase)
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            interval: Time interval (e.g., '1d', '1h', '1m')
            sources: Whether to include sources ('true' or 'false')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the exchange metrics data
        """
        url = f"{self.BASE_URL}/analytics.v2/exchange_metrics"
        params = {
            "exchange": exchange,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval,
            "sources": sources,
            "page_size": page_size
        }
        
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
    
    def get_lending_events(self,
                         blockchain: str,
                         protocol: str,
                         user_address: Optional[str] = None,
                         live: str = "false",
                         tx_hash: Optional[str] = None,
                         asset: Optional[str] = None,
                         event_type: Optional[str] = None,
                         block_number: Optional[int] = None,
                         start_block: Optional[int] = None,
                         end_block: Optional[int] = None,
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         sort: str = "desc",
                         page_size: int = 1000) -> pd.DataFrame:
        """
        Get events data from DeFi lending protocols.
        
        Args:
            blockchain: Blockchain name (e.g., 'ethereum')
            protocol: Protocol code (e.g., 'aav2' for Aave V2)
            user_address: Filter by user address
            live: Whether to include live events ('true' or 'false')
            tx_hash: Filter by transaction hash
            asset: Filter by asset (e.g., 'weth')
            event_type: Filter by event type (e.g., 'borrow', 'deposit', 'repay')
            block_number: Filter by specific block number
            start_block: Start block number
            end_block: End block number
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            sort: Sort order ('asc' or 'desc')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the lending events data
        """
        url = f"{self.EU_BASE_URL}/lending.v1/events"
        params = {
            "blockchain": blockchain,
            "protocol": protocol,
            "sort": sort,
            "page_size": page_size,
            "live": live
        }
        
        # Add optional parameters if provided
        if user_address:
            params["user_address"] = user_address
        if tx_hash:
            params["tx_hash"] = tx_hash
        if asset:
            params["asset"] = asset
        if event_type:
            params["type"] = event_type
        if block_number:
            params["block_number"] = block_number
        if start_block:
            params["start_block"] = start_block
        if end_block:
            params["end_block"] = end_block
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
            
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
    
    def get_liquidity_events(self,
                           blockchain: str,
                           protocol: str,
                           pool_address: Optional[str] = None,
                           pool_contains: Optional[str] = None,
                           block_number: Optional[int] = None,
                           user_addresses: Optional[List[str]] = None,
                           live: str = "false",
                           tx_hash: Optional[str] = None,
                           start_block: Optional[int] = None,
                           end_block: Optional[int] = None,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           sort: str = "desc",
                           event_type: Optional[str] = None,
                           page_size: int = 500) -> pd.DataFrame:
        """
        Get events data from DeFi liquidity pools.
        
        Args:
            blockchain: Blockchain name (e.g., 'ethereum')
            protocol: Protocol code (e.g., 'usp2' for Uniswap V2)
            pool_address: Filter by specific pool address
            pool_contains: Filter pools containing a specific token (e.g., 'weth')
            block_number: Filter by specific block number
            user_addresses: Filter by list of user addresses
            live: Whether to include live events ('true' or 'false')
            tx_hash: Filter by transaction hash
            start_block: Start block number
            end_block: End block number
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            sort: Sort order ('asc' or 'desc')
            event_type: Filter by event type (e.g., 'mint', 'burn', 'swap')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the liquidity events data
        """
        url = f"{self.EU_BASE_URL}/liquidity.v1/events"
        params = {
            "blockchain": blockchain,
            "protocol": protocol,
            "sort": sort,
            "page_size": page_size,
            "live": live
        }
        
        # Add optional parameters if provided
        if pool_address:
            params["pool_address"] = pool_address
        if pool_contains:
            params["pool_contains"] = pool_contains
        if block_number:
            params["block_number"] = block_number
        if user_addresses:
            params["user_addresses"] = ','.join(user_addresses) if isinstance(user_addresses, list) else user_addresses
        if tx_hash:
            params["tx_hash"] = tx_hash
        if start_block:
            params["start_block"] = start_block
        if end_block:
            params["end_block"] = end_block
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if event_type:
            params["type"] = event_type
            
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
    
    def get_trades(self, 
                  exchange: str, 
                  instrument_class: str, 
                  pair: str,
                  start_time: Optional[str] = None,
                  end_time: Optional[str] = None,
                  sort: str = "desc",
                  page_size: int = 100) -> pd.DataFrame:
        """
        Get raw trade data for a specific trading pair.
        
        Args:
            exchange: Exchange code (e.g., 'cbse' for Coinbase)
            instrument_class: Instrument class (e.g., 'spot')
            pair: Trading pair (e.g., 'btc-usd')
            start_time: Start time in ISO 8601 format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO 8601 format (e.g., '2023-12-31T23:59:59Z')
            sort: Sort order ('asc' or 'desc')
            page_size: Number of records per page
            
        Returns:
            DataFrame containing the trade data
        """
        url = f"{self.BASE_URL}/trades.v1/exchanges/{exchange}/{instrument_class}/{pair}/trades"
        params = {
            "sort": sort,
            "page_size": page_size
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
            
        try:
            result_df = pd.DataFrame()
            data = self._make_request(url, params)
            
            if 'data' not in data:
                print("No data returned.")
                return result_df
                
            result_df = pd.DataFrame(data['data'])
            
            # Handle pagination with continuation token
            while 'next_url' in data and data['next_url'] is not None:
                next_url = data['next_url']
                data = self._make_request(next_url, {})
                if 'data' in data and data['data']:
                    result_df = pd.concat([result_df, pd.DataFrame(data['data'])], ignore_index=True)
                else:
                    break
                    
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return pd.DataFrame()
            
    def __enter__(self):
        """Context manager entry method."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        pass


# Example usage:
if __name__ == "__main__":
    # Configuration
    api_key = "YOUR_API_KEY"
    exchange = "cbse"
    instrument_class = "spot"
    pair = "btc-usd"
    start_time = "2023-01-01T00:00:00Z"
    end_time = "2023-12-31T23:59:59Z"
    
    # Using the class directly
    kaiko = KaikoAPI(api_key=api_key)
    
    # Get OHLCV data
    ohlcv_df = kaiko.get_ohlcv(
        exchange=exchange,
        instrument_class=instrument_class,
        pair=pair,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    # Get VWAP data
    vwap_df = kaiko.get_vwap(
        exchange=exchange,
        instrument_class=instrument_class,
        pair=pair,
        start_time=start_time,
        end_time=end_time,
        interval="1d"
    )
    
    # Get asset metrics data
    asset_metrics_df = kaiko.get_asset_metrics(
        asset="agix",
        start_time="2023-01-01T00:00:00Z",
        end_time="2023-12-31T23:59:59Z",
        interval="1d",
        sources="false"
    )
    
    # Get exchange metrics data
    exchange_metrics_df = kaiko.get_exchange_metrics(
        exchange="cbse",
        start_time="2023-01-01T00:00:00Z",
        end_time="2023-01-02T00:00:00Z",
        interval="1h",
        sources="false"
    )
    
    # Get lending events data (for DeFi protocols)
    lending_df = kaiko.get_lending_events(
        blockchain="ethereum",
        protocol="aav2",
        asset="weth",
        event_type="borrow",
        start_time="2023-01-01T00:00:00Z",
        sort="desc",
        page_size=1000
    )
    
    # Get liquidity events data (for DEXs)
    liquidity_df = kaiko.get_liquidity_events(
        blockchain="ethereum",
        protocol="usp2",
        pool_contains="weth",
        event_type="burn",
        start_time="2023-01-01T00:00:00Z",
        sort="asc",
        page_size=500
    )
    
    # Using as a context manager
    with KaikoAPI(api_key=api_key) as kaiko_api:
        df = kaiko_api.get_ohlcv(
            exchange=exchange,
            instrument_class=instrument_class,
            pair=pair,
            start_time=start_time,
            end_time=end_time
        )
