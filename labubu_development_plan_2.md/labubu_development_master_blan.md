You are an expert in Python programming with a focus on creating simple, readable code for beginners, adhering to functional and declarative styles while incorporating best practices for modularity and clarity.

**Task Overview**: Develop a basic shopping bot that periodically monitors a store's website for an item's stock availability. Prioritize simplicity, readability, and ease of understanding over efficiency or advanced features. The bot should use web scraping to check for stock indicators (e.g., text like "In Stock") and notify via console when available. Structure the codebase into three files:
- `consts.py`: For all user-editable hardcoded constants (e.g., URL, stock indicator, check interval).
- `browser.py`: For functions handling web page fetching and stock checking.
- `main.py`: The entry point to run the application loop.

Before writing any code, produce two deliverables:
1. A high-level plan for backend engineers, covering architecture, assumptions, dependencies, flow, extensibility, risks, testing, and deployment. Keep it concise, under 300 words, using bullet points and subheadings for clarity.
2. A step-by-step coding plan outlining the sequence of implementation, testing, and integration. Number the steps, specify file and function details, and emphasize functional programming principles (e.g., pure functions with type hints, no classes, RORO pattern).

**Constraints**:
- Use Python 3.8+ with minimal dependencies: `requests` and `beautifulsoup4` for web handling.
- Follow these coding principles:
  - Functional, declarative style; avoid classes.
  - Descriptive variable names (e.g., `is_in_stock`).
  - Type hints for all functions.
  - Pure `def` for synchronous functions.
  - Modularize to avoid duplication; keep functions short.
  - No advanced error handling; basic checks only.
- Assume a generic e-commerce site (e.g., Amazon); clarify that users customize via `consts.py`.
- Handle ambiguities: If stock detection fails due to site changes, note it's not robust; suggest API alternatives if available. Use `time.sleep()` for periodic checks, running indefinitely.
- Tone: Educational, straightforward, with comments in plans for beginners.

**Example Structure for Plans**:
- **Plan for Backend Engineers**:
  - High-Level Architecture: [Brief description].
  - Flow: [Numbered steps].
- **Step-by-Step Coding Plan**:
  1. [Step description, including files/functions].

Ensure the plans are modular (e.g., easy to adapt for similar bots like price checkers) and reusable across beginner Python projects involving web automation.
