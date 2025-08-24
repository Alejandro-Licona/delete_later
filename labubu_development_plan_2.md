### Plan for Backend Engineers

This shopping bot is designed as a beginner-friendly Python script that periodically checks a store's website for item stock availability. It prioritizes readability and simplicity over efficiency, scalability, or production-grade features (e.g., no error handling beyond basics, no async concurrency, no database). The bot will use web scraping to inspect the page content for stock indicators (e.g., searching for text like "In Stock" or parsing specific HTML elements).

#### High-Level Architecture
- **Modular Structure**: Split into minimal files for clarity:
  - `consts.py`: Holds all user-configurable constants (e.g., URL, item selectors, check interval). Users edit this file only for customization.
  - `browser.py`: Contains pure functions for web interactions (fetching page content and checking stock).
  - `main.py`: The entry point that runs the application loop, importing from the other files.
- **Dependencies**: Use standard libraries where possible; external ones limited to `requests` for HTTP and `beautifulsoup4` for parsing (install via `pip install requests beautifulsoup4`).
- **Flow**:
  1. Load constants from `consts.py`.
  2. In a loop: Fetch the webpage, parse it, check for stock indicators.
  3. If in stock, notify (e.g., print to console; extendable to email/SMS later).
  4. Sleep for a configurable interval (e.g., 5 minutes) before rechecking.
- **Assumptions**:
  - Target store: Example uses a generic e-commerce site (e.g., Amazon product page); user customizes via consts.
  - Stock check: Simple string search or CSS selector for "In Stock" text. Not robust to site changes.
  - Periodic check: Use `time.sleep()` for simplicity; runs indefinitely until stopped (Ctrl+C).
  - No authentication, proxies, or anti-bot measures; this is for educational purposes.
- **Extensibility**: Engineers can add features like logging, email notifications, or multiple items without changing core structure.
- **Risks**: Web scraping may violate terms of service; advise using APIs if available. Frequent checks could lead to IP bans.
- **Testing**: Manual runs; no automated tests for beginner focus.
- **Deployment**: Run as a local script (e.g., `python main.py`); for always-on, suggest cron jobs or simple servers, but not implemented here.

This keeps the codebase under 100 lines total, with short functions and clear comments.

### Step-by-Step Coding Plan

We'll follow this plan sequentially with the coding agent. Each step builds one part, tests it minimally, and integrates. Use functional style: pure functions, type hints with basics (no Pydantic for simplicity), descriptive names. Code in Python 3.8+.

1. **Set Up Project Structure**:
   - Create a directory: `shopping_bot/`.
   - Inside: Create empty files `consts.py`, `browser.py`, `main.py`.
   - Install dependencies: Run `pip install requests beautifulsoup4` (assume agent handles this).

2. **Implement consts.py**:
   - Define constants as global variables (e.g., `STORE_URL: str = "https://example.com/product"`, `STOCK_INDICATOR: str = "In Stock"`, `CHECK_INTERVAL_SECONDS: int = 300`).
   - Add comments explaining each for user editing.
   - Include optional: `USER_AGENT: str` for headers to mimic browser.

3. **Implement browser.py**:
   - Import `requests` and `bs4` (BeautifulSoup).
   - Define function `fetch_page(url: str) -> str`: Use requests.get() with headers; return HTML text if successful, else empty string.
   - Define function `check_stock(html: str, indicator: str) -> bool`: Parse with BeautifulSoup; search for indicator text (e.g., soup.find(string=indicator)); return True if found.
   - Keep functions pure: Receive inputs, return outputs, no side effects.

4. **Implement main.py**:
   - Import from `consts` and `browser`.
   - Define main loop in `if __name__ == "__main__":`.
   - In loop: Call fetch_page, then check_stock; if True, print "Item in stock!"; else "Out of stock.".
   - Use `import time` and `time.sleep(CHECK_INTERVAL_SECONDS)` between checks.
   - Add simple print statements for status.

5. **Test and Refine**:
   - Run `python main.py` manually.
   - Verify: Checks run periodically, detects stock correctly (test with a known in-stock page).
   - If issues: Debug prints in functions; adjust selector in consts if needed.
   - Final touches: Add docstrings/comments for readability.
