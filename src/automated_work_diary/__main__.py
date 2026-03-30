from playwright.sync_api import sync_playwright, Playwright, Page, Request
from sentence_transformers import SentenceTransformer
from automated_work_diary.cluster import update, generate_diary, get_page_summary, get_clustering_info, TIMESTAMP_FMT
import chromadb
from datetime import datetime


CHROMA_DB_PATH = "./browser_memory_db"

def listen_for_events(playwright: Playwright):
    """
    Starts a blocking playwright loop which continuously listens for new loaded pages and
    stores textual context for diary entries. This acts as the main entry function for the package.
    
    Args:
        playwright (Playwright): The playwright instance to use for browser automation.
    Returns:
        None
    """
    
    chromium = playwright.chromium
    browser = chromium.connect_over_cdp("http://localhost:9222") # connect to the browser over CDP
    context = browser.contexts[0]
    
    # initialize the sentence transformer for embedding page content
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # initialize the chroma client for storing browser events
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # get or create the collections for storing browser events and clusters
    collection = chroma_client.get_or_create_collection(name="browser_events")
    clusters_collection = chroma_client.get_or_create_collection(name="browser_clusters")
    
    update_request_count = 0 # used for tracking incremental printing of cluster info
    
    # because I track the total time spent on the page, 
    # we process a page after another page is loaded.
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
            # right now, the text fed into the sentence transformer for embedding 
            # is a combination of title, url, cleaned page content, and a brief AI page summary
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
            
            # encode a single event
            embed_vector = embedder.encode(embed_text, convert_to_numpy=True)
            
            # make an incremental update adding an event embed to the text embed collection, and potentially the centroid collection    
            update(collection, clusters_collection, embed_vector, embed_mdata, embed_text)
            
            # periodically print some clustering info about current cluster
            # this displays how many embeds/page events belong to each cluster
            if update_request_count % 5 == 0:
                get_clustering_info(collection)
            
            update_request_count += 1
            
        # this is a rudimentary check to avoid processing the new tab page
        if page.url == "chrome://new-tab-page/":
            return
        
        # update the last load timestamp and page state
        last_load_timestamp = current_time
        last_url = page.url
        last_title = page.title()
        last_content = page.content()
        last_page_summary = get_page_summary(last_title, last_content) # TODO: make this non-blocking on the main thread
        
        
        
    # this attaches a playwright listener to new tabs 
    # that triggers when the tab goes to a new page
    def on_page_created(page: Page):
        print("New tab found")
        page.on("load", handle_page_load)

    # this attaches a listener for the whole window itself
    context.on("page", on_page_created)
    
    # track pre-existing tabs/pages
    for page in context.pages:
        on_page_created(page)
        
    try:
        while True:
            # events aren't that rapid so don't make excessive calls
            context.pages[0].wait_for_timeout(1000)
            
    except KeyboardInterrupt:
        # exitting triggers diary creation
        # TODO: make diary generation periodic and automatic
        print("exiting...")
        generate_diary(collection)
        context.close()
        browser.close()
        
with sync_playwright() as playwright:
    listen_for_events(playwright)