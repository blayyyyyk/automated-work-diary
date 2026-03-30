# Automated Work Diary

## Setup Guide
**Prerequisites:** * Python >= 3.12
* Google Chrome installed on your machine
* [Ollama](https://ollama.com/) installed on your machine

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```
2. Install Playwright browsers:
```bash
playwright install chromium
```
3. Start the local LLM:
Ensure Ollama is running, then pull the required model:
```bash
ollama run llama3:8b
```
4. Launch Chrome with Remote Debugging:
Before running the tracker, completely close all instances of Chrome. Then, launch Chrome from your terminal with the remote debugging port open:
    - Mac: `/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222`
    - Windows: `"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222`

## Usage

Once Chrome and Ollama are running, start the tracker:
```bash
python -m automated_work_diary
```