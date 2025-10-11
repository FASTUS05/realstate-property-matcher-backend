# config.py - Add this file for easy tuning
class MatchingConfig:
    # Search parameters (tune these for speed vs accuracy)
    TEXT_TOP_K = 20          # Reduced for speed
    IMAGE_TOP_K = 20         # Reduced for speed  
    FINAL_CANDIDATES = 30    # Reduced for speed
    
    # Scoring weights
    TEXT_WEIGHT = 0.45
    IMAGE_WEIGHT = 0.35
    STRUCTURED_WEIGHT = 0.20
    
    # Performance settings
    MAX_WORKERS = 4          # Adjust based on your CPU cores
    IMAGE_TIMEOUT = 5        # Reduced timeout for image downloads
    MAX_IMAGES_PER_LISTING = 2  # Limit images processed
    
    # For even faster results, use these aggressive settings:
    @classmethod
    def fast_mode(cls):
        cls.TEXT_TOP_K = 10
        cls.IMAGE_TOP_K = 10
        cls.FINAL_CANDIDATES = 15
        cls.IMAGE_WEIGHT = 0.2  # Reduce image weight since it's slower
        cls.TEXT_WEIGHT = 0.6   # Increase text weight
        cls.STRUCTURED_WEIGHT = 0.2