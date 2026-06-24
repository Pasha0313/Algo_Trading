import requests
import logging
import time
from requests.exceptions import HTTPError

def make_request_with_retries(url, max_retries=3, timeout=10):
    for attempt in range(max_retries):
        try:
           response = requests.get(url, timeout=timeout)
           response.raise_for_status()  # Check if the request was successful
           return response
        except HTTPError as e:
           if attempt < max_retries - 1:
               time.sleep(2 ** attempt)  # Exponential backoff
           else:
               raise
        except requests.RequestException as e:
            # Handle other request exceptions (e.g., connection errors)
            raise

def Debug_function():
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
    
#try:
#    account_info = client.get_account()
#    print("Account info:", account_info)
#except Exception as e:
#    print(f"An error occurred: {e}")

#try:
#    system_status = client.get_system_status()
#    print("System status:", system_status)
#except Exception as e:
#    print(f"An error occurred: {e}")
        
#Debug_function()

#url = "https://testnet.binancefuture.com/api/v1/someendpoint"
#response = make_request_with_retries(url, timeout=10)
#data = response.json()
#response = requests.get(url, timeout=30)  # Increase timeout to 30 seconds

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')    

 #logging.info("Starting the application")    