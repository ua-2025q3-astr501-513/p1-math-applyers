"""
Script to download SOHO comet data and background sequences.

Usage:
    python 01_download_data.py --output data/raw --years 2005 2010 --max-comets 50
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection import (
    CometDataCollector,
    SOHOImageDownloader,
    BackgroundDownloader
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download SOHO comet data and images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for data'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=list(range(2005, 2022)),
        help='Years to scrape (e.g., 2005 2010 2015)'
    )
    
    parser.add_argument(
        '--max-comets',
        type=int,
        default=50,
        help='Maximum number of comets to download images for'
    )
    
    parser.add_argument(
        '--max-backgrounds',
        type=int,
        default=20,
        help='Number of background sequences to download'
    )
    
    parser.add_argument(
        '--window-hours',
        type=int,
        default=3,
        help='Time window around observations (±hours)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=50,
        help='Maximum images per sequence'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup directories
    output_dir = Path(args.output)
    comet_images_dir = output_dir / 'comet_images'
    comet_positions_dir = output_dir / 'comet_positions'
    background_images_dir = output_dir / 'background_images'
    
    for directory in [comet_images_dir, comet_positions_dir, background_images_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("COMET-SEE DATA COLLECTION")
    logger.info("="*60)
    
    # Step 1: Scrape comet catalog
    logger.info("\n[1/4] Scraping comet catalog...")
    collector = CometDataCollector()
    comet_df = collector.scrape_multiple_years(args.years)
    
    # Save catalog
    catalog_file = comet_positions_dir / 'comet_catalog.csv'
    comet_df.to_csv(catalog_file, index=False)
    logger.info(f"Saved catalog to {catalog_file}")
    
    # Step 2: Download position files
    logger.info("\n[2/4] Downloading position files...")
    comets_with_times = []
    
    for idx, row in comet_df.head(args.max_comets).iterrows():
        times = collector.download_position_file(
            row['position_file_url'],
            row['soho_number'],
            str(comet_positions_dir)
        )
        
        if times and len(times) > 0:
            comets_with_times.append({
                'soho_number': row['soho_number'],
                'obs_time': times[0],
                'num_observations': len(times)
            })
    
    times_df = pd.DataFrame(comets_with_times)
    times_file = comet_positions_dir / 'comets_with_times.csv'
    times_df.to_csv(times_file, index=False)
    logger.info(f"Got times for {len(times_df)} comets")
    
    # Step 3: Download comet images
    logger.info("\n[3/4] Downloading comet images...")
    downloader = SOHOImageDownloader()
    
    downloaded_count = 0
    for idx, row in times_df.iterrows():
        files = downloader.download_for_comet(
            row['soho_number'],
            row['obs_time'],
            str(comet_images_dir),
            window_hours=args.window_hours,
            max_images=args.max_images
        )
        
        if files:
            downloaded_count += 1
    
    logger.info(f"Downloaded images for {downloaded_count} comets")
    
    # Step 4: Download background sequences
    logger.info("\n[4/4] Downloading background sequences...")
    
    # Get comet dates to avoid
    comet_dates = set()
    for obs_time in times_df['obs_time']:
        try:
            dt = pd.to_datetime(obs_time)
            comet_dates.add(dt.date())
        except:
            continue
    
    bg_downloader = BackgroundDownloader()
    bg_count = bg_downloader.download_backgrounds(
        num_sequences=args.max_backgrounds,
        comet_dates=comet_dates,
        output_dir=str(background_images_dir),
        start_year=min(args.years),
        end_year=max(args.years),
        window_hours=args.window_hours,
        max_images=args.max_images
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total comets in catalog: {len(comet_df)}")
    logger.info(f"Comets with observation times: {len(times_df)}")
    logger.info(f"Comet sequences downloaded: {downloaded_count}")
    logger.info(f"Background sequences downloaded: {bg_count}")
    logger.info(f"\nData saved to: {output_dir}")


if __name__ == '__main__':
    main()