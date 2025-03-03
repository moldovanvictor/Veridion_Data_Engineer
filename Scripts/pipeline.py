import pandas as pd
import numpy as np
import re
import logging
import os
from typing import Tuple, Dict, List, Optional, Any


# Set up logging
def setup_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_integration.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("data_integration")


# Data Loading
def load_dataset(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load a CSV dataset with error handling.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame or None if loading fails
    """
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Set default parameters for CSV reading
        default_params = {
            'quotechar': '"',
            'escapechar': '\\',
            'on_bad_lines': 'warn',
            'low_memory': False
        }

        # Combine default parameters with any provided kwargs
        params = {**default_params, **kwargs}

        # Load the dataset
        df = pd.read_csv(file_path, **params)

        # Basic validation
        if df.empty:
            raise ValueError(f"File {file_path} loaded but contains no data")

        return df

    except Exception as e:
        logging.error(f"Error loading dataset {file_path}: {str(e)}")
        return None


def load_all_datasets(base_path: str = "../datasets/") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all three datasets with appropriate parameters.

    Args:
        base_path: Base directory containing dataset files

    Returns:
        Tuple of (website_df, google_df, fb_df)
    """
    try:
        # Define file paths
        website_path = os.path.join(base_path, "website_dataset.csv")
        google_path = os.path.join(base_path, "google_dataset.csv")
        fb_path = os.path.join(base_path, "facebook_dataset.csv")

        # Load each dataset with appropriate parameters
        website_df = load_dataset(website_path, sep=";")
        google_df = load_dataset(google_path)
        fb_df = load_dataset(fb_path)

        # Validate all datasets were loaded
        if website_df is None or google_df is None or fb_df is None:
            logging.error("Failed to load one or more datasets")
            return None, None, None

        return website_df, google_df, fb_df

    except Exception as e:
        logging.error(f"Error in load_all_datasets: {str(e)}")
        return None, None, None


# Data Exploration
def explore_dataset(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Perform basic exploration of a dataset.

    Args:
        df: DataFrame to explore
        name: Name of the dataset for reporting

    Returns:
        Dictionary of exploration results
    """
    try:
        # Get domain column name (different across datasets)
        domain_col = 'root_domain' if 'root_domain' in df.columns else 'domain'

        # Gather basic statistics
        stats = {
            'name': name,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_values': df.isna().sum().sort_values(ascending=False).head(5).to_dict(),
            'domain_duplicates': df[domain_col].duplicated().sum()
        }

        return stats

    except Exception as e:
        logging.error(f"Error exploring dataset {name}: {str(e)}")
        return {'name': name, 'error': str(e)}


def explore_all_datasets(website_df: pd.DataFrame, google_df: pd.DataFrame, fb_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Explore all three datasets and return consolidated results.

    Args:
        website_df: Website dataset
        google_df: Google dataset
        fb_df: Facebook dataset

    Returns:
        Dictionary of exploration results for all datasets
    """
    logging.info("Starting data exploration...")

    results = {
        'facebook': explore_dataset(fb_df, 'Facebook'),
        'google': explore_dataset(google_df, 'Google'),
        'website': explore_dataset(website_df, 'Website')
    }

    for dataset, stats in results.items():
        logging.info(f"{dataset.capitalize()} Dataset: {stats.get('rows')} rows, {stats.get('columns')} columns")
        logging.info(f"Domain duplicates: {stats.get('domain_duplicates')}")

    return results


# Data Cleaning
def clean_domain(domain) -> Optional[str]:
    """
    Standardize domain names by removing prefixes and trailing content.

    Args:
        domain: Domain string to clean

    Returns:
        Cleaned domain or None if input is invalid
    """
    if not isinstance(domain, str):
        return None

    try:
        # Strip whitespace
        domain = domain.strip().lower()

        # Remove http://, https://, www. prefixes and trailing slashes
        domain = re.sub(r'^(https?://)?(www\.)?', '', domain)
        domain = re.sub(r'/.*$', '', domain)

        # Validate result
        if not domain or '.' not in domain:
            return None

        return domain

    except Exception as e:
        logging.debug(f"Error cleaning domain {domain}: {str(e)}")
        return None


def standardize_phone(phone) -> Optional[str]:
    """
    Standardize phone numbers to digits-only format.

    Args:
        phone: Phone number to standardize

    Returns:
        Standardized phone number or None if invalid
    """
    if not isinstance(phone, str) and not isinstance(phone, int):
        return None

    try:
        phone_str = str(phone)

        # Remove non-digit characters
        digits_only = re.sub(r'\D', '', phone_str)

        # If no digits, return None
        if not digits_only:
            return None

        # Validate length (reasonable phone numbers have at least 7 digits)
        if len(digits_only) < 7:
            return None

        return digits_only

    except Exception as e:
        logging.debug(f"Error standardizing phone {phone}: {str(e)}")
        return None


def standardize_category(category) -> Optional[str]:
    """
    Standardize category strings.

    Args:
        category: Category string to standardize

    Returns:
        Standardized category or None if invalid
    """
    if not isinstance(category, str):
        return None

    try:
        # Convert to lowercase
        category = category.lower()

        # Remove some common words that don't add value
        category = re.sub(r'\b(and|&|in|of|the|services)\b', ' ', category)

        # Replace multiple spaces with single space
        category = re.sub(r'\s+', ' ', category)

        category = category.strip()

        # If category is empty after cleaning, return None
        if not category:
            return None

        return category

    except Exception as e:
        logging.debug(f"Error standardizing category {category}: {str(e)}")
        return None


def clean_dataset(df: pd.DataFrame, source_name: str, domain_col: str = 'domain',phone_col: str = 'phone', category_col: str = 'categories') -> pd.DataFrame:
    """
    Clean a dataset by standardizing domains, phones, and categories.

    Args:
        df: DataFrame to clean
        source_name: Name of data source
        domain_col: Column name for domain
        phone_col: Column name for phone
        category_col: Column name for category

    Returns:
        Cleaned DataFrame
    """
    try:
        logging.info(f"Cleaning {source_name} dataset...")

        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()

        # Apply cleaning functions
        cleaned_df['domain_clean'] = cleaned_df[domain_col].apply(clean_domain)

        if phone_col in cleaned_df.columns:
            cleaned_df['phone_clean'] = cleaned_df[phone_col].apply(standardize_phone)
        else:
            cleaned_df['phone_clean'] = None

        if category_col in cleaned_df.columns:
            cleaned_df['categories_clean'] = cleaned_df[category_col].apply(standardize_category)
        else:
            cleaned_df['categories_clean'] = None

        # Add source column
        cleaned_df['source'] = source_name

        # Remove rows with null domains after cleaning
        orig_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['domain_clean'])
        removed_count = orig_count - len(cleaned_df)

        logging.info(
            f"Removed {removed_count} rows ({removed_count / orig_count:.1%}) with invalid domains from {source_name} dataset")

        return cleaned_df

    except Exception as e:
        logging.error(f"Error cleaning {source_name} dataset: {str(e)}")
        raise


def clean_all_datasets(website_df: pd.DataFrame, google_df: pd.DataFrame, fb_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean all three datasets.

    Args:
        website_df: Website dataset
        google_df: Google dataset
        fb_df: Facebook dataset

    Returns:
        Tuple of cleaned DataFrames
    """
    logging.info("Starting data cleaning process...")

    # Clean Facebook data
    fb_cleaned = clean_dataset(
        fb_df,
        'facebook',
        domain_col='domain',
        phone_col='phone',
        category_col='categories'
    )

    # Clean Google data
    google_cleaned = clean_dataset(
        google_df,
        'google',
        domain_col='domain',
        phone_col='phone',
        category_col='category'
    )

    # Clean Website data
    website_cleaned = clean_dataset(
        website_df,
        'website',
        domain_col='root_domain',
        phone_col='phone',
        category_col='s_category'
    )

    logging.info(
        f"After cleaning - Facebook: {fb_cleaned.shape[0]} rows, "
        f"Google: {google_cleaned.shape[0]} rows, "
        f"Website: {website_cleaned.shape[0]} rows"
    )

    return fb_cleaned, google_cleaned, website_cleaned


# Domain Overlap Analysis
def analyze_domain_overlap(fb_df: pd.DataFrame, google_df: pd.DataFrame, website_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze domain overlap between datasets.

    Args:
        fb_df: Facebook dataset
        google_df: Google dataset
        website_df: Website dataset

    Returns:
        Dictionary of overlap statistics
    """
    try:
        # Extract sets of domains
        fb_domains = set(fb_df['domain_clean'])
        google_domains = set(google_df['domain_clean'])
        website_domains = set(website_df['domain_clean'])

        # Calculate overlaps
        fb_google_overlap = fb_domains.intersection(google_domains)
        fb_website_overlap = fb_domains.intersection(website_domains)
        google_website_overlap = google_domains.intersection(website_domains)
        all_overlap = fb_domains.intersection(google_domains, website_domains)

        # Generate report
        overlap_stats = {
            'facebook_domains': len(fb_domains),
            'google_domains': len(google_domains),
            'website_domains': len(website_domains),
            'facebook_google_overlap': len(fb_google_overlap),
            'facebook_website_overlap': len(fb_website_overlap),
            'google_website_overlap': len(google_website_overlap),
            'all_three_overlap': len(all_overlap),
            'facebook_google_pct': len(fb_google_overlap) / len(fb_domains) * 100 if fb_domains else 0,
            'facebook_website_pct': len(fb_website_overlap) / len(fb_domains) * 100 if fb_domains else 0,
            'google_website_pct': len(google_website_overlap) / len(google_domains) * 100 if google_domains else 0,
            'all_three_pct': len(all_overlap) / (len(fb_domains) or 1) * 100
        }

        # Log results
        logging.info(f"Facebook unique domains: {len(fb_domains)}")
        logging.info(f"Google unique domains: {len(google_domains)}")
        logging.info(f"Website unique domains: {len(website_domains)}")
        logging.info(f"Facebook-Google overlap: {len(fb_google_overlap)} domains")
        logging.info(f"Facebook-Website overlap: {len(fb_website_overlap)} domains")
        logging.info(f"Google-Website overlap: {len(google_website_overlap)} domains")
        logging.info(f"Overlap in all three datasets: {len(all_overlap)} domains")

        return overlap_stats

    except Exception as e:
        logging.error(f"Error in analyze_domain_overlap: {str(e)}")
        return {'error': str(e)}


# Data Integration
def prepare_datasets_for_joining(fb_df: pd.DataFrame, google_df: pd.DataFrame, website_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare datasets for joining by selecting and renaming relevant columns.

    Args:
        fb_df: Facebook dataset
        google_df: Google dataset
        website_df: Website dataset

    Returns:
        Tuple of prepared DataFrames
    """
    try:
        logging.info("Preparing datasets for joining...")

        # Select relevant columns from Facebook
        fb_selected = fb_df[['domain_clean', 'name', 'categories_clean', 'address', 'city',
                             'country_name', 'region_name', 'phone_clean', 'email', 'source']]
        fb_selected = fb_selected.rename(columns={
            'name': 'company_name',
            'categories_clean': 'category',
            'country_name': 'country',
            'region_name': 'region'
        })

        # Select relevant columns from Google
        google_selected = google_df[['domain_clean', 'name', 'categories_clean', 'address', 'city',
                                     'country_name', 'region_name', 'phone_clean', 'source']]
        google_selected = google_selected.rename(columns={
            'name': 'company_name',
            'categories_clean': 'category',
            'country_name': 'country',
            'region_name': 'region'
        })
        # Add email column to match structure
        google_selected['email'] = np.nan

        # Select relevant columns from Website
        website_selected = website_df[['domain_clean', 'site_name', 'legal_name', 'categories_clean',
                                       'main_city', 'main_country', 'main_region', 'phone_clean', 'source']]
        website_selected = website_selected.rename(columns={
            'site_name': 'company_name',
            'categories_clean': 'category',
            'main_city': 'city',
            'main_country': 'country',
            'main_region': 'region'
        })
        # Fill company_name with legal_name if site_name is missing
        website_selected['company_name'] = website_selected['company_name'].fillna(website_selected['legal_name'])
        website_selected = website_selected.drop(columns=['legal_name'])
        # Add missing columns
        website_selected['address'] = np.nan
        website_selected['email'] = np.nan

        # Validate prepared datasets
        expected_columns = ['domain_clean', 'company_name', 'category', 'address', 'city',
                            'country', 'region', 'phone_clean', 'email', 'source']

        for df, name in [(fb_selected, 'Facebook'), (google_selected, 'Google'), (website_selected, 'Website')]:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                logging.warning(f"{name} dataset is missing expected columns: {missing_cols}")

        logging.info("Datasets prepared for joining")
        return fb_selected, google_selected, website_selected

    except Exception as e:
        logging.error(f"Error in prepare_datasets_for_joining: {str(e)}")
        raise


def define_source_priorities() -> Dict[str, List[str]]:
    """
    Define source priorities for resolving conflicts.

    Returns:
        Dictionary of field-specific source priorities
    """
    # Based on assumed data quality and completeness per field
    return {
        'company_name': ['facebook', 'website', 'google'],
        'category': ['google', 'website', 'facebook'],
        'address': ['google', 'facebook', 'website'],
        'city': ['google', 'facebook', 'website'],
        'country': ['google', 'facebook', 'website'],
        'region': ['google', 'facebook', 'website'],
        'phone_clean': ['facebook', 'google', 'website'],
        'email': ['facebook', 'website', 'google']
    }


def resolve_conflicts(group: pd.DataFrame, field: str, priority_list: List[str]) -> Any:
    """
    Resolve conflicts between data sources based on priority.

    Args:
        group: Group of rows for a single domain
        field: Field to resolve
        priority_list: List of sources in priority order

    Returns:
        Resolved value
    """
    try:
        # Get values that are not NaN
        valid_values = group[group[field].notna()]

        if valid_values.empty:
            return np.nan

        # If only one valid value, return it
        if len(valid_values) == 1:
            return valid_values[field].iloc[0]

        # If all values are the same, return that value
        unique_values = valid_values[field].unique()
        if len(unique_values) == 1:
            return unique_values[0]

        # Otherwise, prioritize based on source
        for source in priority_list:
            source_values = valid_values[valid_values['source'] == source]
            if not source_values.empty:
                return source_values[field].iloc[0]

        # Fallback to first value if no priority source found
        return valid_values[field].iloc[0]

    except Exception as e:
        logging.debug(f"Error resolving conflicts for field {field}: {str(e)}")
        return np.nan


def aggregate_company_data(group: pd.DataFrame, priorities: Dict[str, List[str]]) -> pd.Series:
    """
    Aggregate data for a single company across sources.

    Args:
        group: Group of rows for a single domain
        priorities: Dictionary of field-specific source priorities

    Returns:
        Series with aggregated company data
    """
    try:
        return pd.Series({
            'domain': group['domain_clean'].iloc[0],
            'company_name': resolve_conflicts(group, 'company_name', priorities.get('company_name', [])),
            'category': resolve_conflicts(group, 'category', priorities.get('category', [])),
            'address': resolve_conflicts(group, 'address', priorities.get('address', [])),
            'city': resolve_conflicts(group, 'city', priorities.get('city', [])),
            'country': resolve_conflicts(group, 'country', priorities.get('country', [])),
            'region': resolve_conflicts(group, 'region', priorities.get('region', [])),
            'phone': resolve_conflicts(group, 'phone_clean', priorities.get('phone_clean', [])),
            'email': resolve_conflicts(group, 'email', priorities.get('email', [])),
            'data_sources': ','.join(group['source'].unique())
        })

    except Exception as e:
        logging.error(f"Error aggregating company data: {str(e)}")
        # Return a Series with domain to avoid breaking the groupby
        return pd.Series({'domain': group['domain_clean'].iloc[0], 'error': str(e)})


def integrate_datasets(fb_selected: pd.DataFrame, google_selected: pd.DataFrame,website_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate datasets into a unified company dataset.

    Args:
        fb_selected: Prepared Facebook dataset
        google_selected: Prepared Google dataset
        website_selected: Prepared Website dataset

    Returns:
        Integrated DataFrame
    """
    try:
        logging.info("Integrating datasets...")

        # Combine datasets
        combined_df = pd.concat([fb_selected, google_selected, website_selected], ignore_index=True)

        # Check for duplicated domains within sources (should be cleaned already)
        for source in combined_df['source'].unique():
            source_df = combined_df[combined_df['source'] == source]
            dup_count = source_df['domain_clean'].duplicated().sum()
            if dup_count > 0:
                logging.warning(f"Found {dup_count} duplicated domains in {source} dataset after preparation")

        # Count occurrences per domain
        domain_counts = combined_df['domain_clean'].value_counts()
        unique_domains = domain_counts.index.tolist()

        logging.info(f"Total unique domains after combining: {len(unique_domains)}")
        logging.info(f"Domains appearing in multiple sources: {sum(domain_counts > 1)}")

        # Get source priorities
        source_priorities = define_source_priorities()

        # Create final dataset
        logging.info("Creating final unified dataset...")
        final_df = combined_df.groupby('domain_clean').apply(
            lambda x: aggregate_company_data(x, source_priorities)
        ).reset_index(drop=True)

        # Check for any rows with errors
        if 'error' in final_df.columns:
            error_count = final_df['error'].notna().sum()
            if error_count > 0:
                logging.warning(f"Encountered errors when integrating {error_count} companies")
                final_df = final_df[final_df['error'].isna()].drop(columns=['error'])

        logging.info(f"Final dataset created with {final_df.shape[0]} companies")
        return final_df

    except Exception as e:
        logging.error(f"Error in integrate_datasets: {str(e)}")
        raise


def save_final_dataset(final_df: pd.DataFrame, output_file: str = 'unified_company_dataset.csv'):
    """
    Save the final dataset to a CSV file.

    Args:
        final_df: Final integrated dataset
        output_file: File to save dataset
    """
    try:
        final_df.to_csv(output_file, index=False)
        logging.info(f"Final dataset saved as '{output_file}'")

    except Exception as e:
        logging.error(f"Error saving final dataset: {str(e)}")


def validate_final_dataset(final_df: pd.DataFrame) -> bool:
    """
    Validate the final dataset to ensure quality.

    Args:
        final_df: Final integrated dataset

    Returns:
        True if validation passes, False otherwise
    """
    try:
        logging.info("Validating final dataset...")

        # Check required columns
        required_columns = ['domain', 'company_name', 'data_sources']
        missing_columns = set(required_columns) - set(final_df.columns)
        if missing_columns:
            logging.error(f"Final dataset is missing required columns: {missing_columns}")
            return False

        # Check for duplicate domains
        dup_count = final_df['domain'].duplicated().sum()
        if dup_count > 0:
            logging.error(f"Final dataset contains {dup_count} duplicate domains")
            return False

        # Check for empty domains
        null_domains = final_df['domain'].isna().sum()
        if null_domains > 0:
            logging.error(f"Final dataset contains {null_domains} rows with null domains")
            return False

        # Check for empty company names
        null_companies = final_df['company_name'].isna().sum()
        if null_companies > len(final_df) * 0.5:  # If more than 50% are missing
            logging.warning(f"Final dataset contains {null_companies} rows with null company names")

        # Check data completeness
        completeness = final_df.notna().mean() * 100
        low_completeness = completeness[completeness < 30].index.tolist()
        if low_completeness:
            logging.warning(f"These fields have low completeness (<30%): {low_completeness}")

        logging.info("Final dataset validation passed")
        return True

    except Exception as e:
        logging.error(f"Error validating final dataset: {str(e)}")
        return False


# Main function
def main(datasets_path: str = "../datasets/",output_dir: str = "./",output_file: str = "unified_company_dataset.csv"):
    """
    Main function to execute the entire data integration pipeline.

    Args:
        datasets_path: Path to directory containing input datasets
        output_dir: Directory to save output files
        output_file: Name of output dataset file
    """
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data integration pipeline...")

    try:
        # 1. Load datasets
        logger.info("Loading datasets from {}".format(datasets_path))
        website_df, google_df, fb_df = load_all_datasets(datasets_path)

        if website_df is None or google_df is None or fb_df is None:
            raise ValueError("Failed to load one or more datasets")

        # 2. Explore datasets
        logger.info("Exploring datasets...")
        exploration_results = explore_all_datasets(website_df, google_df, fb_df)

        # 3. Clean datasets
        logger.info("Cleaning datasets...")
        fb_cleaned, google_cleaned, website_cleaned = clean_all_datasets(website_df, google_df, fb_df)

        # 4. Analyze domain overlap
        logger.info("Analyzing domain overlap...")
        overlap_stats = analyze_domain_overlap(fb_cleaned, google_cleaned, website_cleaned)

        # 5. Prepare datasets for joining
        logger.info("Preparing datasets for joining...")
        fb_prepared, google_prepared, website_prepared = prepare_datasets_for_joining(
            fb_cleaned, google_cleaned, website_cleaned
        )

        # 6. Integrate datasets
        logger.info("Integrating datasets...")
        final_df = integrate_datasets(fb_prepared, google_prepared, website_prepared)

        # 7. Validate final dataset
        logger.info("Validating final dataset...")
        validation_passed = validate_final_dataset(final_df)
        if not validation_passed:
            logger.warning("Final dataset validation failed, but continuing with process")

        # 8. Save final dataset
        logger.info("Saving final dataset...")
        output_path = os.path.join(output_dir, output_file)
        save_final_dataset(final_df, output_path)

        logger.info("Data integration pipeline completed successfully")
        return final_df

    except Exception as e:
        logger.error(f"Error in data integration pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run data integration pipeline for company data')
    parser.add_argument('--datasets', type=str, default='../datasets/',
                        help='Path to directory containing input datasets')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Directory to save output files')
    parser.add_argument('--output-file', type=str, default='unified_company_dataset.csv',
                        help='Name of output dataset file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Set up logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    # Run the pipeline
    main(datasets_path=args.datasets, output_dir=args.output_dir, output_file=args.output_file)
