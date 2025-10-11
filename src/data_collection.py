"""
Data collection module for COMET-SEE project.

This module handles:
- Scraping comet catalogs from Sungrazer Project
- Downloading comet position files
- Fetching SOHO/LASCO C3 FITS images
- Downloading background sequences
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta
import os
import time
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CometDataCollector:
    """Collects SOHO comet data from Sungrazer Project."""
    
    def __init__(self, base_url: str = "https://sungrazer.nrl.navy.mil"):
        """
        Initialize the data collector.
        
        Args:
            base_url: Base URL for Sungrazer Project
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def scrape_comet_table(self, year: int) -> List[Dict]:
        """
        Scrape comet table for a given year.
        
        Args:
            year: Year to scrape
            
        Returns:
            List of dictionaries containing comet information
        """
        url = f'{self.base_url}/comets_table_{year}'
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                logger.warning(f"No table found for {year}")
                return []
            
            comets = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    designation_cell = cols[1]
                    link = designation_cell.find('a')
                    
                    if link and 'href' in link.attrs:
                        href = link['href']
                        # Extract SOHO number from link
                        match = re.search(r'soho(\d+)_xy\.txt', href)
                        if match:
                            soho_num = match.group(1)
                            comets.append({
                                'soho_number': f'SOHO-{soho_num}',
                                'year': year,
                                'position_file_url': self.base_url + href
                            })
            
            logger.info(f"Found {len(comets)} comets from {year}")
            return comets
            
        except Exception as e:
            logger.error(f"Error scraping {year}: {e}")
            return []
    
    def scrape_multiple_years(self, years: List[int]) -> pd.DataFrame:
        """
        Scrape comet data for multiple years.
        
        Args:
            years: List of years to scrape
            
        Returns:
            DataFrame with all comet data
        """
        all_comets = []
        
        for year in years:
            comets = self.scrape_comet_table(year)
            all_comets.extend(comets)
            time.sleep(1)  # Be polite to the server
        
        df = pd.DataFrame(all_comets)
        logger.info(f"Total comets found: {len(df)}")
        return df
    
    def download_position_file(
        self, 
        url: str, 
        soho_number: str, 
        output_dir: str
    ) -> Optional[List[str]]:
        """
        Download and parse comet position file.
        
        Args:
            url: URL to position file
            soho_number: SOHO identifier
            output_dir: Directory to save file
            
        Returns:
            List of observation timestamps
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Save file
            filename = Path(output_dir) / f"{soho_number}_positions.txt"
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(response.text)
            
            # Parse observation times
            lines = response.text.strip().split('\n')
            times = []
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    date_time = f"{parts[0]} {parts[1]}"
                    times.append(date_time)
            
            logger.info(f"{soho_number}: Downloaded {len(times)} observation times")
            return times
            
        except Exception as e:
            logger.error(f"Error downloading {soho_number}: {e}")
            return None


class SOHOImageDownloader:
    """Downloads SOHO/LASCO C3 images for comet observations."""
    
    def __init__(self, instrument: str = 'LASCO', detector: str = 'C3'):
        """
        Initialize the image downloader.
        
        Args:
            instrument: SOHO instrument name
            detector: Detector name
        """
        self.instrument = instrument
        self.detector = detector
    
    def download_images_for_time(
        self,
        center_time: datetime,
        window_hours: int = 3,
        output_dir: str = 'data/images',
        max_images: int = 50
    ) -> Optional[List[str]]:
        """
        Download images around a specific time.
        
        Args:
            center_time: Center time for observation
            window_hours: Time window in hours (±)
            output_dir: Directory to save images
            max_images: Maximum number of images to download
            
        Returns:
            List of downloaded file paths
        """
        start = center_time - timedelta(hours=window_hours)
        end = center_time + timedelta(hours=window_hours)
        
        try:
            # Search for images
            result = Fido.search(
                a.Time(start, end),
                a.Instrument(self.instrument),
                a.Detector(self.detector)
            )
            
            if len(result) == 0:
                logger.warning(f"No images found for {center_time}")
                return None
            
            logger.info(f"Found {len(result[0])} images")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Download images
            files = Fido.fetch(
                result[0][:max_images],
                path=output_dir + "/{file}"
            )
            
            logger.info(f"Downloaded {len(files)} images to {output_dir}")
            return files
            
        except Exception as e:
            logger.error(f"Error downloading images: {e}")
            return None
    
    def download_for_comet(
        self,
        soho_number: str,
        obs_time_str: str,
        base_output_dir: str,
        **kwargs
    ) -> Optional[List[str]]:
        """
        Download images for a specific comet observation.
        
        Args:
            soho_number: SOHO comet identifier
            obs_time_str: Observation time string (YYYY-MM-DD HH:MM:SS)
            base_output_dir: Base directory for output
            **kwargs: Additional arguments for download_images_for_time
            
        Returns:
            List of downloaded file paths
        """
        try:
            obs_time = datetime.strptime(obs_time_str, '%Y-%m-%d %H:%M:%S')
            output_dir = Path(base_output_dir) / soho_number
            
            logger.info(f"Downloading images for {soho_number} at {obs_time}")
            
            return self.download_images_for_time(
                center_time=obs_time,
                output_dir=str(output_dir),
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error processing {soho_number}: {e}")
            return None


class BackgroundDownloader:
    """Downloads background sequences without comet activity."""
    
    def __init__(self, instrument: str = 'LASCO', detector: str = 'C3'):
        self.instrument = instrument
        self.detector = detector
        self.downloader = SOHOImageDownloader(instrument, detector)
    
    def generate_safe_dates(
        self,
        num_dates: int,
        start_year: int,
        end_year: int,
        comet_dates: set,
        max_attempts: int = 1000
    ) -> List[datetime]:
        """
        Generate random dates that don't overlap with comet observations.
        
        Args:
            num_dates: Number of dates to generate
            start_year: Start year for random selection
            end_year: End year for random selection
            comet_dates: Set of dates with comet activity
            max_attempts: Maximum attempts to find safe dates
            
        Returns:
            List of safe datetime objects
        """
        safe_dates = []
        attempts = 0
        
        while len(safe_dates) < num_dates and attempts < max_attempts:
            year = np.random.randint(start_year, end_year + 1)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            hour = np.random.choice([0, 6, 12, 18])
            
            try:
                random_date = datetime(year, month, day, hour, 0)
                
                # Check if safe (no comet within ±1 day)
                if random_date.date() not in comet_dates:
                    prev_day = (random_date - timedelta(days=1)).date()
                    next_day = (random_date + timedelta(days=1)).date()
                    
                    if prev_day not in comet_dates and next_day not in comet_dates:
                        safe_dates.append(random_date)
            except:
                pass
            
            attempts += 1
        
        logger.info(f"Generated {len(safe_dates)} safe background dates")
        return safe_dates
    
    def download_backgrounds(
        self,
        num_sequences: int,
        comet_dates: set,
        output_dir: str,
        start_year: int = 2005,
        end_year: int = 2023,
        **kwargs
    ) -> int:
        """
        Download background image sequences.
        
        Args:
            num_sequences: Number of background sequences to download
            comet_dates: Set of dates with comet activity
            output_dir: Base output directory
            start_year: Start year for random selection
            end_year: End year for random selection
            **kwargs: Additional arguments for image download
            
        Returns:
            Number of successfully downloaded sequences
        """
        safe_dates = self.generate_safe_dates(
            num_sequences, start_year, end_year, comet_dates
        )
        
        successful = 0
        
        for i, date_dt in enumerate(safe_dates, 1):
            logger.info(f"[{i}/{len(safe_dates)}] {date_dt.strftime('%Y-%m-%d %H:%M')}")
            
            folder = Path(output_dir) / f"background_{i:02d}"
            
            files = self.downloader.download_images_for_time(
                center_time=date_dt,
                output_dir=str(folder),
                **kwargs
            )
            
            if files:
                successful += 1
            
            time.sleep(2)  # Be polite to the server
        
        logger.info(f"Successfully downloaded {successful} background sequences")
        return successful