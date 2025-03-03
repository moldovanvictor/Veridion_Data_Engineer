# Data Integration Project

This project integrates company data from three different sources (Facebook, Google, and company websites) to create a unified dataset with improved accuracy on key business information.

## Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Data Investigation](#data-investigation)
- [Data Cleaning](#data-cleaning)
- [Integration Process](#integration-process)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## Overview

This project aims to create a comprehensive, accurate dataset of company information by combining data from multiple sources. The integration process involves loading, exploring, cleaning, and merging data to produce a unified dataset with improved accuracy for key fields such as company categories, addresses, phone numbers, and names.

## Data Sources

The project uses three primary data sources:
1. **Facebook** (`facebook_dataset.csv`) - Company data scraped from Facebook pages
2. **Google** (`google_dataset.csv`) - Company information from Google Business listings
3. **Website** (`website_dataset.csv`) - Data extracted directly from company websites

Each source has its strengths and weaknesses in terms of data completeness and accuracy.

## Data Investigation

The data investigation phase consists of several key steps to understand the characteristics and quality of each dataset:

### Basic Exploration
- Examining dataset dimensions (rows/columns)
- Identifying missing values
- Analyzing column data types
- Checking for duplicated domains

```
# Example of exploration code
def explore_dataset(df, name):
    domain_col = 'root_domain' if 'root_domain' in df.columns else 'domain'
    stats = {
        'name': name,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isna().sum().sort_values(ascending=False).head(5).to_dict(),
        'domain_duplicates': df[domain_col].duplicated().sum()
    }
    return stats
```

### Cross-Dataset Analysis
- Calculating domain overlap between sources
- Comparing field completeness across datasets
- Evaluating data format inconsistencies

The exploration revealed:
- Different column naming conventions across datasets
- Varying formats for the same data (e.g., phone numbers, domains)
- Different levels of completeness for key fields
- Significant overlap between datasets, but also unique records in each

## Data Cleaning

The data cleaning process addresses several quality issues found during exploration:

### Domain Standardization
- Removing `http://`, `https://`, and `www.` prefixes
- Eliminating trailing slashes and URL paths
- Standardizing to lowercase

```
def clean_domain(domain):
    if not isinstance(domain, str):
        return None
    
    domain = domain.strip().lower()
    domain = re.sub(r'^(https?://)?(www\.)?', '', domain)
    domain = re.sub(r'/.*$', '', domain)
    
    if not domain or '.' not in domain:
        return None
    
    return domain
```

### Phone Number Standardization
- Converting to digits-only format
- Validating length and format
- Handling international prefixes

### Category Normalization
- Converting to lowercase
- Removing filler words ("and", "&", "the", etc.)
- Standardizing naming conventions

### Address Components
- Standardizing country, region, and city names
- Separating address components where needed
- Handling multilingual entries

## Integration Process

The integration strategy follows these steps:

### 1. Prepare Datasets
- Select relevant columns from each source
- Rename columns to a common schema
- Handle missing values consistently

### 2. Define Source Priorities
Different sources have varying reliability for different fields:
- Company names: Facebook > Website > Google
- Categories: Google > Website > Facebook
- Addresses: Google > Facebook > Website
- Phone numbers: Facebook > Google > Website
- Email addresses: Facebook > Website > Google

### 3. Resolution Logic
When conflicting information exists for the same company:
1. If only one source has a value, use it
2. If multiple sources have the same value, use it
3. If sources disagree, use the value from the highest priority source

```
def resolve_conflicts(group, field, priority_list):
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
```

### 4. Validation
- Checking for duplicate domains in the final dataset
- Verifying data completeness
- Ensuring key fields (domain, company_name) are populated

## Getting Started

### Prerequisites
- Python 3.7+
- pandas
- numpy
- logging

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/data-integration-project.git
cd data-integration-project
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Usage

Run the main integration script:

```
python pipeline.py --datasets /path/to/datasets/ --output-dir ./output/ --output-file unified_companies.csv
```

Optional arguments:
- `--datasets`: Path to directory containing input datasets (default: '../datasets/')
- `--output-dir`: Directory to save output files (default: './')
- `--output-file`: Name of output dataset file (default: 'unified_company_dataset.csv')
- `--debug`: Enable debug logging

## Project Structure

```
data-integration-project/
├── datasets/               # Input datasets
│   ├── facebook_dataset.csv
│   ├── google_dataset.csv
│   └── website_dataset.csv
├── Scripts/               
│   ├── pipeline.py         # Main integration script
├── output/                 # Output directory
│   └── unified_company_dataset.csv
├── data_integration.log    # Logging output
├── README.md               # Project documentation
└── requirements.txt        # Required packages
```

## Results

The final integrated dataset contains:
- One row per unique company domain
- Best available company name, category, address, and contact information
- Record of which sources contributed to each company profile
- Significantly improved completeness compared to any single source

The integration process yields a dataset with:
- Higher accuracy through cross-validation
- Better field completeness than any individual source
- Standardized formatting for consistent analysis
- Clear lineage showing data origin