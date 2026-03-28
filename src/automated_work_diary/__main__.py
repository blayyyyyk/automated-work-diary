from playwright.sync_api import sync_playwright, Playwright, Page, Request
from sentence_transformers import SentenceTransformer
from automated_work_diary.cluster import update, summarize_clusters, get_page_summary, get_clustering_info, TIMESTAMP_FMT
import chromadb
from datetime import datetime


CHROMA_DB_PATH = "./browser_memory_db"
    
def run(playwright: Playwright):
    chromium = playwright.chromium
    browser = chromium.connect_over_cdp("http://localhost:9222")
    context = browser.contexts[0]
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="browser_events")
    clusters_collection = chroma_client.get_or_create_collection(name="browser_clusters")
    
    update_request_count = 0
    
    last_load_timestamp = datetime.now()
    last_url = None
    last_title = None
    last_content = None
    last_page_summary = None
    
    docs = []
    mdatas = []
    
    def update_docs(text: str, mdata: str):
        docs.append(text)
        mdatas.append(mdata)
        if len(docs) >= 5000:
            docs
    
    def handle_page_load(page: Page):
        nonlocal last_url, last_title, last_content, last_load_timestamp, last_page_summary, embedder, collection, clusters_collection, update_request_count
        
        current_time = datetime.now()
        time_spent = current_time - last_load_timestamp
        
        if all([last_url, last_title, last_content, last_page_summary]):
            embed_text = f"Title: {last_title}\nURL: {last_url}\nContent: {last_content}\nAI Summary: {last_page_summary}"
            embed_mdata = {
                "url": last_url,
                "title": last_title,
                "time_spent": time_spent.total_seconds(),
                "time_start": last_load_timestamp.strftime(TIMESTAMP_FMT),
                "time_end": current_time.strftime(TIMESTAMP_FMT),
                "content": last_content,
                "summary": last_page_summary
            }
            
            embed_vector = embedder.encode(embed_text, convert_to_numpy=True)
                
            update(collection, clusters_collection, embed_vector, embed_mdata, embed_text)
            if update_request_count % 5 == 0:
                get_clustering_info(collection)
            
            update_request_count += 1
            
        if page.url == "chrome://new-tab-page/":
            return
        
        last_load_timestamp = current_time
        last_url = page.url
        last_title = page.title()
        last_content = page.content()
        last_page_summary = get_page_summary(last_title, last_content)
        
        
        
    
    def on_page_created(page: Page):
        print("New tab found")
        # page.on("request", handle_page_request)
        page.on("load", handle_page_load)
    
    context.on("page", on_page_created)
    
    # track pre-existing pages
    for page in context.pages:
        on_page_created(page)
        
    try:
        while True:
            context.pages[0].wait_for_timeout(1000)
            
    except KeyboardInterrupt:
        print("exiting...")
        summarize_clusters(collection)
        context.close()
        browser.close()
        
        
    
if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
