import time 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Batch processor worker started...")
    logger.info("Wait batch processing task...")

    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()